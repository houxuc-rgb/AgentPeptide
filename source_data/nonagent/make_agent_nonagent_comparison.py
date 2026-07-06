from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SOURCE_DIR = Path(__file__).resolve().parents[1]
NONAGENT_DIR = Path(__file__).resolve().parent
OUT_DIR = NONAGENT_DIR

POOL_LABELS = {
    "sol": "Soluble",
    "nf": "Non-fouling",
    "hem": "Hemolysis",
}

POOL_COLORS = {
    "sol": "#5B7FCA",
    "nf": "#33B5A5",
    "hem": "#D9544D",
}

METHOD_COLORS = {
    "Origin": "#767676",
    "Agent": "#484878",
    "Non-agent": "#2E9E44",
}


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def score(sol: float, nf: float, hem: float) -> float:
    return sol + nf + (1.0 - hem)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    var = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(var / len(values))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_matched_rows() -> list[dict]:
    selected_rows = read_csv(NONAGENT_DIR / "selected_peptides_for_optimizer_comparison.csv")
    nonagent_rows = {
        row["start_id"]: row
        for row in read_csv(NONAGENT_DIR / "nonagent_greedy_summary.csv")
    }

    origins = {
        pool: read_json(SOURCE_DIR / "origin" / f"sampled_{pool}.json")
        for pool in POOL_LABELS
    }
    refined = {
        pool: read_json(SOURCE_DIR / "refined" / f"sampled_{pool}.json")
        for pool in POOL_LABELS
    }

    rows = []
    for selected in selected_rows:
        start_id = selected["start_id"]
        pool = selected["source_pool"]
        origin_idx = int(selected["original_idx"])
        origin = origins[pool][origin_idx]
        agent = refined[pool][origin_idx]
        nonagent = nonagent_rows[start_id]

        if origin["sequence"] != selected["sequence"]:
            raise ValueError(
                f"Selected/origin mismatch for {start_id}: "
                f"{selected['sequence']} != {origin['sequence']}"
            )
        if nonagent["original_sequence"] != origin["sequence"]:
            raise ValueError(
                f"Non-agent/origin mismatch for {start_id}: "
                f"{nonagent['original_sequence']} != {origin['sequence']}"
            )
        if int(origin["idx"]) != int(agent["idx"]):
            raise ValueError(f"Origin/refined idx mismatch for {start_id}")

        origin_sol = float(origin["sol"])
        origin_nf = float(origin["nf"])
        origin_hem = float(origin["hem"])
        agent_sol = float(agent["sol"])
        agent_nf = float(agent["nf"])
        agent_hem = float(agent["hem"])
        non_sol = float(nonagent["selected_solubility"])
        non_nf = float(nonagent["selected_non_fouling"])
        non_hem = float(nonagent["selected_hemolysis"])

        origin_s = score(origin_sol, origin_nf, origin_hem)
        agent_s = score(agent_sol, agent_nf, agent_hem)
        non_s = score(non_sol, non_nf, non_hem)

        rows.append(
            {
                "start_id": start_id,
                "source_pool": pool,
                "source_label": POOL_LABELS[pool],
                "score_band": selected["score_band"],
                "origin_idx": origin_idx,
                "origin_sequence": origin["sequence"],
                "agent_sequence": agent["sequence"],
                "nonagent_sequence": nonagent["selected_sequence"],
                "origin_sol": origin_sol,
                "origin_nf": origin_nf,
                "origin_hem": origin_hem,
                "origin_S": origin_s,
                "agent_sol": agent_sol,
                "agent_nf": agent_nf,
                "agent_hem": agent_hem,
                "agent_S": agent_s,
                "nonagent_sol": non_sol,
                "nonagent_nf": non_nf,
                "nonagent_hem": non_hem,
                "nonagent_S": non_s,
                "agent_delta_S": agent_s - origin_s,
                "nonagent_delta_S": non_s - origin_s,
                "nonagent_minus_agent_S": non_s - agent_s,
                "agent_delta_sol": agent_sol - origin_sol,
                "agent_delta_nf": agent_nf - origin_nf,
                "agent_delta_antihem": origin_hem - agent_hem,
                "nonagent_delta_sol": non_sol - origin_sol,
                "nonagent_delta_nf": non_nf - origin_nf,
                "nonagent_delta_antihem": origin_hem - non_hem,
                "agent_pass_all": agent_sol >= 0.5 and agent_nf >= 0.5 and agent_hem < 0.5,
                "nonagent_pass_all": non_sol >= 0.5 and non_nf >= 0.5 and non_hem < 0.5,
            }
        )

    return rows


def build_summaries(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_method_property = []
    property_map = [
        ("Solubility", "delta_sol"),
        ("Non-fouling", "delta_nf"),
        ("Antihemolysis", "delta_antihem"),
    ]
    for method, prefix in [("Agent", "agent"), ("Non-agent", "nonagent")]:
        for prop_label, suffix in property_map:
            vals = [row[f"{prefix}_{suffix}"] for row in rows]
            by_method_property.append(
                {
                    "method": method,
                    "property": prop_label,
                    "n": len(vals),
                    "mean_desired_delta": mean(vals),
                    "sem_desired_delta": sem(vals),
                    "improved_count": sum(value > 0 for value in vals),
                    "worsened_count": sum(value < 0 for value in vals),
                    "unchanged_count": sum(value == 0 for value in vals),
                }
            )

    by_pool = []
    for pool in ["sol", "nf", "hem"]:
        pool_rows = [row for row in rows if row["source_pool"] == pool]
        for method, key in [("Agent", "agent_delta_S"), ("Non-agent", "nonagent_delta_S")]:
            vals = [row[key] for row in pool_rows]
            by_pool.append(
                {
                    "source_pool": pool,
                    "source_label": POOL_LABELS[pool],
                    "method": method,
                    "n": len(vals),
                    "mean_delta_S": mean(vals),
                    "sem_delta_S": sem(vals),
                    "improved_count": sum(value > 0 for value in vals),
                }
            )
    for method, key in [("Agent", "agent_delta_S"), ("Non-agent", "nonagent_delta_S")]:
        vals = [row[key] for row in rows]
        by_pool.append(
            {
                "source_pool": "all",
                "source_label": "All",
                "method": method,
                "n": len(vals),
                "mean_delta_S": mean(vals),
                "sem_delta_S": sem(vals),
                "improved_count": sum(value > 0 for value in vals),
            }
        )
    return by_method_property, by_pool


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_comparison(rows: list[dict], property_summary: list[dict], pool_summary: list[dict]) -> None:
    sorted_rows = sorted(rows, key=lambda r: r["nonagent_minus_agent_S"], reverse=True)
    x = list(range(len(sorted_rows)))

    fig = plt.figure(figsize=(7.2, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.38, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # Panel a: per-start composite-score changes for identical origins.
    for i, row in enumerate(sorted_rows):
        pool_color = POOL_COLORS[row["source_pool"]]
        ax_a.plot([i, i], [row["agent_S"], row["nonagent_S"]], color="#D0D0D0", lw=0.8, zorder=1)
        ax_a.scatter(i, row["origin_S"], s=13, color=METHOD_COLORS["Origin"], alpha=0.75, zorder=3)
        ax_a.scatter(
            i,
            row["agent_S"],
            s=21,
            color=METHOD_COLORS["Agent"],
            edgecolor=pool_color,
            linewidth=0.55,
            zorder=4,
        )
        ax_a.scatter(
            i,
            row["nonagent_S"],
            s=24,
            marker="^",
            color=METHOD_COLORS["Non-agent"],
            edgecolor=pool_color,
            linewidth=0.55,
            zorder=5,
        )

    ax_a.axhline(mean([row["origin_S"] for row in rows]), color=METHOD_COLORS["Origin"], lw=0.9, ls="--")
    ax_a.axhline(mean([row["agent_S"] for row in rows]), color=METHOD_COLORS["Agent"], lw=0.9, ls="--")
    ax_a.axhline(mean([row["nonagent_S"] for row in rows]), color=METHOD_COLORS["Non-agent"], lw=0.9, ls="--")
    ax_a.set_ylabel("Composite score S")
    ax_a.set_xlabel("Representative origin peptides sorted by non-agent advantage")
    ax_a.set_xlim(-0.8, len(sorted_rows) - 0.2)
    ax_a.set_ylim(1.84, 2.08)
    ax_a.set_xticks([])
    ax_a.grid(axis="y", color="#E8E8E8", lw=0.55)
    add_panel_label(ax_a, "a")
    ax_a.text(
        0.02,
        0.98,
        "Same 24 origin peptides\nNon-agent higher S: 24/24",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D8D8D8", linewidth=0.6),
    )
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS["Origin"], markeredgewidth=0, label="Origin"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS["Agent"], markeredgewidth=0, label="Agent"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor=METHOD_COLORS["Non-agent"], markeredgewidth=0, label="Non-agent"),
        ],
        loc="lower right",
        handletextpad=0.4,
        borderpad=0.2,
        labelspacing=0.25,
    )

    # Panel b: mean property-direction changes.
    props = ["Solubility", "Non-fouling", "Antihemolysis"]
    prop_labels = ["Solubility", "Non-fouling", "1 - hemolysis"]
    centers = [0, 1, 2]
    width = 0.32
    summary_lookup = {(row["method"], row["property"]): row for row in property_summary}
    for method, offset in [("Agent", -width / 2), ("Non-agent", width / 2)]:
        means = [summary_lookup[(method, prop)]["mean_desired_delta"] for prop in props]
        errs = [summary_lookup[(method, prop)]["sem_desired_delta"] for prop in props]
        ax_b.bar(
            [c + offset for c in centers],
            means,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            yerr=errs,
            capsize=2.5,
            error_kw={"lw": 0.75, "capthick": 0.75},
            label=method,
        )
    ax_b.axhline(0, color="#4D4D4D", lw=0.8)
    ax_b.set_xticks(centers)
    ax_b.set_xticklabels(prop_labels, rotation=25, ha="right")
    ax_b.set_ylabel("Mean desired-property change")
    ax_b.set_ylim(-0.02, 0.055)
    ax_b.grid(axis="y", color="#E8E8E8", lw=0.55)
    add_panel_label(ax_b, "b")
    ax_b.legend(loc="upper left", handlelength=1.1)

    # Panel c: source-pool contribution to composite-score gain.
    labels = ["Soluble", "Non-fouling", "Hemolysis", "All"]
    pool_keys = ["sol", "nf", "hem", "all"]
    centers = [0, 1, 2, 3]
    pool_lookup = {(row["source_pool"], row["method"]): row for row in pool_summary}
    for method, offset in [("Agent", -width / 2), ("Non-agent", width / 2)]:
        means = [pool_lookup[(pool, method)]["mean_delta_S"] for pool in pool_keys]
        errs = [pool_lookup[(pool, method)]["sem_delta_S"] for pool in pool_keys]
        ax_c.bar(
            [c + offset for c in centers],
            means,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            yerr=errs,
            capsize=2.5,
            error_kw={"lw": 0.75, "capthick": 0.75},
        )
    ax_c.axhline(0, color="#4D4D4D", lw=0.8)
    ax_c.set_xticks(centers)
    ax_c.set_xticklabels(labels, rotation=25, ha="right")
    ax_c.set_ylabel("Mean change in S")
    ax_c.set_ylim(-0.035, 0.105)
    ax_c.grid(axis="y", color="#E8E8E8", lw=0.55)
    add_panel_label(ax_c, "c")
    ax_c.text(
        0.04,
        0.96,
        "Greedy local scan improves\nall source-pool strata",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )

    for ax in [ax_a, ax_b, ax_c]:
        ax.tick_params(length=3, width=0.7)

    fig.suptitle(
        "Non-agent greedy refinement is a strong one-step local-search baseline",
        fontsize=8.5,
        y=1.02,
    )

    stem = OUT_DIR / "agent_vs_nonagent_greedy_comparison"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_qa_notes(rows: list[dict], property_summary: list[dict], pool_summary: list[dict]) -> None:
    agent_delta = [row["agent_delta_S"] for row in rows]
    non_delta = [row["nonagent_delta_S"] for row in rows]
    non_minus_agent = [row["nonagent_minus_agent_S"] for row in rows]

    notes = f"""# Agent vs non-agent greedy comparison figure QA

Core conclusion: On the same 24 origin peptides, exhaustive non-agent one-step greedy refinement is a stronger local-search baseline than the current conservative agent refinement under the composite predictor score.

Evidence chain:
- Panel a: per-origin composite scores for origin, agent-refined, and non-agent greedy outputs.
- Panel b: mean desired-direction property changes for solubility, non-fouling, and antihemolysis.
- Panel c: mean composite-score change by source pool and overall.

Archetype: quantitative grid with one hero comparison panel and two summary evidence panels.

Backend: Python/matplotlib only.

Export contract:
- Primary editable vector: `agent_vs_nonagent_greedy_comparison.svg`
- Secondary vector: `agent_vs_nonagent_greedy_comparison.pdf`
- Raster previews: `agent_vs_nonagent_greedy_comparison.png`, `agent_vs_nonagent_greedy_comparison.tiff`
- Source data: `matched_agent_nonagent_source.csv`, `summary_by_method_property.csv`, `summary_by_pool.csv`

Data integrity:
- Origin values were read from the sampled origin JSON files by source pool and `idx`.
- Agent-refined values were read from the corresponding refined JSON files by the same source pool and `idx`.
- Non-agent values were read from `nonagent_greedy_summary.csv` and matched by `start_id` plus origin sequence.
- Existing JSON files were read only; no source JSON was modified.

Summary:
- n = {len(rows)} representative origin peptides.
- Mean agent delta S = {mean(agent_delta):.6f}; improved count = {sum(value > 0 for value in agent_delta)}/{len(rows)}.
- Mean non-agent delta S = {mean(non_delta):.6f}; improved count = {sum(value > 0 for value in non_delta)}/{len(rows)}.
- Mean non-agent minus agent S = {mean(non_minus_agent):.6f}; non-agent higher count = {sum(value > 0 for value in non_minus_agent)}/{len(rows)}.

Review risks:
- The non-agent scan is edit-depth matched but not prediction-call matched because it exhaustively evaluates local single mutants.
- The comparison covers 24 representative starts, not the full 300-peptide pool.
- All results are predictor-defined and require experimental validation before biological claims.
"""
    (OUT_DIR / "qa_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_matched_rows()
    property_summary, pool_summary = build_summaries(rows)

    source_fields = [
        "start_id",
        "source_pool",
        "source_label",
        "score_band",
        "origin_idx",
        "origin_sequence",
        "agent_sequence",
        "nonagent_sequence",
        "origin_sol",
        "origin_nf",
        "origin_hem",
        "origin_S",
        "agent_sol",
        "agent_nf",
        "agent_hem",
        "agent_S",
        "nonagent_sol",
        "nonagent_nf",
        "nonagent_hem",
        "nonagent_S",
        "agent_delta_S",
        "nonagent_delta_S",
        "nonagent_minus_agent_S",
        "agent_delta_sol",
        "agent_delta_nf",
        "agent_delta_antihem",
        "nonagent_delta_sol",
        "nonagent_delta_nf",
        "nonagent_delta_antihem",
        "agent_pass_all",
        "nonagent_pass_all",
    ]
    write_csv(OUT_DIR / "matched_agent_nonagent_source.csv", rows, source_fields)
    write_csv(
        OUT_DIR / "summary_by_method_property.csv",
        property_summary,
        [
            "method",
            "property",
            "n",
            "mean_desired_delta",
            "sem_desired_delta",
            "improved_count",
            "worsened_count",
            "unchanged_count",
        ],
    )
    write_csv(
        OUT_DIR / "summary_by_pool.csv",
        pool_summary,
        ["source_pool", "source_label", "method", "n", "mean_delta_S", "sem_delta_S", "improved_count"],
    )
    plot_comparison(rows, property_summary, pool_summary)
    write_qa_notes(rows, property_summary, pool_summary)


if __name__ == "__main__":
    main()
