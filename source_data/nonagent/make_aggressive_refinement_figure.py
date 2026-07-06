from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageStat


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.65
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    color: str


@dataclass(frozen=True)
class Paths:
    out_dir: Path
    nonagent_dir: Path
    base_dir: Path


METHODS = (
    MethodSpec("conservative", "Conservative agent", "#7884B4"),
    MethodSpec("greedy", "Greedy control", "#767676"),
    MethodSpec("aggressive", "Aggressive agent", "#C76F82"),
)
METHOD_BY_KEY = {method.key: method for method in METHODS}
METHOD_ORDER = [method.key for method in METHODS]

POOL_FILES = {
    "sol": "sampled_sol.json",
    "nf": "sampled_nf.json",
    "hem": "sampled_hem.json",
}
POOL_LABELS = {
    "sol": "Soluble",
    "nf": "Non-fouling",
    "hem": "Hemolysis",
    "overall": "All",
}
SUMMARY_POOLS = ("sol", "nf", "hem", "overall")

RECORD_FIELDS = [
    "method",
    "method_label",
    "source_pool",
    "source_label",
    "start_id",
    "original_sequence",
    "final_sequence",
    "original_length",
    "final_length",
    "delta_length",
    "length_changed",
    "original_solubility",
    "final_solubility",
    "delta_solubility",
    "original_non_fouling",
    "final_non_fouling",
    "delta_non_fouling",
    "original_hemolysis",
    "final_hemolysis",
    "delta_antihemolysis",
    "original_score",
    "final_score",
    "delta_score",
    "score_improved",
    "threshold_fail_final",
]

SUMMARY_FIELDS = [
    "method",
    "method_label",
    "source_pool",
    "source_label",
    "n",
    "original_score_mean",
    "final_score_mean",
    "delta_score_mean",
    "delta_score_sem",
    "delta_solubility_mean",
    "delta_solubility_sem",
    "delta_non_fouling_mean",
    "delta_non_fouling_sem",
    "delta_antihemolysis_mean",
    "delta_antihemolysis_sem",
    "score_improved_count",
    "threshold_fail_final_count",
    "length_changed_count",
    "delta_length_mean",
]


def project_paths() -> Paths:
    out_dir = Path(__file__).resolve().parent
    return Paths(out_dir=out_dir, nonagent_dir=out_dir, base_dir=out_dir.parent)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value) -> float:
    return float(value)


def composite_score(solubility: float, non_fouling: float, hemolysis: float) -> float:
    return solubility + non_fouling + (1.0 - hemolysis)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def sem(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def normalized_record(
    *,
    method: str,
    source_pool: str,
    start_id: str,
    original_sequence: str,
    final_sequence: str,
    original_solubility: float,
    final_solubility: float,
    original_non_fouling: float,
    final_non_fouling: float,
    original_hemolysis: float,
    final_hemolysis: float,
) -> dict:
    method_spec = METHOD_BY_KEY[method]
    original_score = composite_score(
        original_solubility, original_non_fouling, original_hemolysis
    )
    final_score = composite_score(final_solubility, final_non_fouling, final_hemolysis)
    original_length = len(original_sequence)
    final_length = len(final_sequence)

    return {
        "method": method,
        "method_label": method_spec.label,
        "source_pool": source_pool,
        "source_label": POOL_LABELS[source_pool],
        "start_id": start_id,
        "original_sequence": original_sequence,
        "final_sequence": final_sequence,
        "original_length": original_length,
        "final_length": final_length,
        "delta_length": final_length - original_length,
        "length_changed": int(original_length != final_length),
        "original_solubility": original_solubility,
        "final_solubility": final_solubility,
        "delta_solubility": final_solubility - original_solubility,
        "original_non_fouling": original_non_fouling,
        "final_non_fouling": final_non_fouling,
        "delta_non_fouling": final_non_fouling - original_non_fouling,
        "original_hemolysis": original_hemolysis,
        "final_hemolysis": final_hemolysis,
        "delta_antihemolysis": original_hemolysis - final_hemolysis,
        "original_score": original_score,
        "final_score": final_score,
        "delta_score": final_score - original_score,
        "score_improved": int(final_score > original_score),
        "threshold_fail_final": int(
            final_solubility < 0.5 or final_non_fouling < 0.5 or final_hemolysis >= 0.5
        ),
    }


def indexed_json_by_pool(folder: Path) -> dict[str, dict[int, dict]]:
    indexed = {}
    for pool, filename in POOL_FILES.items():
        indexed[pool] = {
            int(row["idx"]): row for row in read_json_rows(folder / filename)
        }
    return indexed


def load_conservative_records(paths: Paths, selected_rows: list[dict[str, str]]) -> list[dict]:
    original_by_pool = indexed_json_by_pool(paths.base_dir / "origin")
    refined_by_pool = indexed_json_by_pool(paths.base_dir / "refined")

    records = []
    for selected in selected_rows:
        pool = selected["source_pool"]
        idx = int(selected["original_idx"])
        original = original_by_pool[pool][idx]
        refined = refined_by_pool[pool][idx]
        records.append(
            normalized_record(
                method="conservative",
                source_pool=pool,
                start_id=selected["start_id"],
                original_sequence=original["sequence"],
                final_sequence=refined["sequence"],
                original_solubility=as_float(original["sol"]),
                final_solubility=as_float(refined["sol"]),
                original_non_fouling=as_float(original["nf"]),
                final_non_fouling=as_float(refined["nf"]),
                original_hemolysis=as_float(original["hem"]),
                final_hemolysis=as_float(refined["hem"]),
            )
        )
    return records


def load_greedy_records(paths: Paths) -> list[dict]:
    records = []
    for row in read_csv_rows(paths.nonagent_dir / "nonagent_greedy_summary.csv"):
        records.append(
            normalized_record(
                method="greedy",
                source_pool=row["source_pool"],
                start_id=row["start_id"],
                original_sequence=row["original_sequence"],
                final_sequence=row["selected_sequence"],
                original_solubility=as_float(row["original_solubility"]),
                final_solubility=as_float(row["selected_solubility"]),
                original_non_fouling=as_float(row["original_non_fouling"]),
                final_non_fouling=as_float(row["selected_non_fouling"]),
                original_hemolysis=as_float(row["original_hemolysis"]),
                final_hemolysis=as_float(row["selected_hemolysis"]),
            )
        )
    return records


def load_aggressive_records(paths: Paths) -> list[dict]:
    records = []
    for row in read_csv_rows(paths.nonagent_dir / "aggressive_refinement_summary.csv"):
        records.append(
            normalized_record(
                method="aggressive",
                source_pool=row["source_pool"],
                start_id=row["start_id"],
                original_sequence=row["original_sequence"],
                final_sequence=row["final_sequence"],
                original_solubility=as_float(row["original_solubility"]),
                final_solubility=as_float(row["final_solubility"]),
                original_non_fouling=as_float(row["original_non_fouling"]),
                final_non_fouling=as_float(row["final_non_fouling"]),
                original_hemolysis=as_float(row["original_hemolysis"]),
                final_hemolysis=as_float(row["final_hemolysis"]),
            )
        )
    return records


def load_all_records(paths: Paths) -> list[dict]:
    selected_rows = read_csv_rows(
        paths.nonagent_dir / "selected_peptides_for_optimizer_comparison.csv"
    )
    return [
        *load_conservative_records(paths, selected_rows),
        *load_greedy_records(paths),
        *load_aggressive_records(paths),
    ]


def summarize_records(records: list[dict]) -> list[dict]:
    summary = []
    for method in METHOD_ORDER:
        method_records = [row for row in records if row["method"] == method]
        for pool in SUMMARY_POOLS:
            rows = (
                method_records
                if pool == "overall"
                else [row for row in method_records if row["source_pool"] == pool]
            )
            summary.append(
                {
                    "method": method,
                    "method_label": METHOD_BY_KEY[method].label,
                    "source_pool": pool,
                    "source_label": POOL_LABELS[pool],
                    "n": len(rows),
                    "original_score_mean": mean(row["original_score"] for row in rows),
                    "final_score_mean": mean(row["final_score"] for row in rows),
                    "delta_score_mean": mean(row["delta_score"] for row in rows),
                    "delta_score_sem": sem(row["delta_score"] for row in rows),
                    "delta_solubility_mean": mean(row["delta_solubility"] for row in rows),
                    "delta_solubility_sem": sem(row["delta_solubility"] for row in rows),
                    "delta_non_fouling_mean": mean(
                        row["delta_non_fouling"] for row in rows
                    ),
                    "delta_non_fouling_sem": sem(
                        row["delta_non_fouling"] for row in rows
                    ),
                    "delta_antihemolysis_mean": mean(
                        row["delta_antihemolysis"] for row in rows
                    ),
                    "delta_antihemolysis_sem": sem(
                        row["delta_antihemolysis"] for row in rows
                    ),
                    "score_improved_count": sum(row["score_improved"] for row in rows),
                    "threshold_fail_final_count": sum(
                        row["threshold_fail_final"] for row in rows
                    ),
                    "length_changed_count": sum(row["length_changed"] for row in rows),
                    "delta_length_mean": mean(row["delta_length"] for row in rows),
                }
            )
    return summary


def head_to_head_records(records: list[dict]) -> list[dict]:
    by_start = {}
    for row in records:
        by_start.setdefault(row["start_id"], {})[row["method"]] = row

    rows = []
    for start_id in sorted(by_start):
        matched = by_start[start_id]
        conservative = matched["conservative"]
        greedy = matched["greedy"]
        aggressive = matched["aggressive"]
        rows.append(
            {
                "start_id": start_id,
                "source_pool": aggressive["source_pool"],
                "conservative_score": conservative["final_score"],
                "greedy_score": greedy["final_score"],
                "aggressive_score": aggressive["final_score"],
                "aggressive_minus_conservative": aggressive["final_score"]
                - conservative["final_score"],
                "aggressive_minus_greedy": aggressive["final_score"]
                - greedy["final_score"],
                "aggressive_gt_conservative": int(
                    aggressive["final_score"] > conservative["final_score"]
                ),
                "aggressive_gt_greedy": int(
                    aggressive["final_score"] > greedy["final_score"]
                ),
            }
        )
    return rows


def export_source_data(paths: Paths, records: list[dict], summary: list[dict], head_to_head: list[dict]) -> None:
    write_csv_rows(
        paths.out_dir / "aggressive_comparison_matched_records.csv",
        records,
        RECORD_FIELDS,
    )
    write_csv_rows(paths.out_dir / "aggressive_summary_stats.csv", summary, SUMMARY_FIELDS)
    write_csv_rows(
        paths.out_dir / "aggressive_head_to_head.csv",
        head_to_head,
        [
            "start_id",
            "source_pool",
            "conservative_score",
            "greedy_score",
            "aggressive_score",
            "aggressive_minus_conservative",
            "aggressive_minus_greedy",
            "aggressive_gt_conservative",
            "aggressive_gt_greedy",
        ],
    )


def save_publication_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def grouped_bar_panel(
    ax: plt.Axes,
    groups: list[str],
    values_by_method: dict[str, list[float]],
    sem_by_method: dict[str, list[float]],
    ylabel: str,
) -> None:
    width = 0.22
    offsets = {"conservative": -width, "greedy": 0.0, "aggressive": width}
    positions = list(range(len(groups)))

    for method in METHOD_ORDER:
        xs = [x + offsets[method] for x in positions]
        spec = METHOD_BY_KEY[method]
        ax.bar(
            xs,
            values_by_method[method],
            width=width,
            color=spec.color,
            edgecolor="#333333",
            linewidth=0.35,
            label=spec.label,
            zorder=3,
        )
        ax.errorbar(
            xs,
            values_by_method[method],
            yerr=sem_by_method[method],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.55,
            capsize=1.8,
            capthick=0.55,
            zorder=4,
        )

    ax.axhline(0, color="#333333", linewidth=0.65, zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.55, zorder=0)


def plot_score_trajectories(ax: plt.Axes, records: list[dict]) -> None:
    lookup = {(row["method"], row["start_id"]): row for row in records}
    starts = sorted({row["start_id"] for row in records})
    x = [0, 1, 2, 3]
    origin_scores = []
    method_scores = {method: [] for method in METHOD_ORDER}

    for index, start_id in enumerate(starts):
        conservative = lookup[("conservative", start_id)]
        values = [
            conservative["original_score"],
            conservative["final_score"],
            lookup[("greedy", start_id)]["final_score"],
            lookup[("aggressive", start_id)]["final_score"],
        ]
        jitter = ((index % 7) - 3) * 0.012
        ax.plot(x, values, color="#BDBDBD", linewidth=0.45, alpha=0.55, zorder=1)
        ax.scatter(
            [value + jitter for value in x],
            values,
            s=9,
            color=[
                "#303030",
                METHOD_BY_KEY["conservative"].color,
                METHOD_BY_KEY["greedy"].color,
                METHOD_BY_KEY["aggressive"].color,
            ],
            edgecolor="white",
            linewidth=0.18,
            zorder=2,
        )
        origin_scores.append(values[0])
        for method in METHOD_ORDER:
            method_scores[method].append(lookup[(method, start_id)]["final_score"])

    means = [
        mean(origin_scores),
        mean(method_scores["conservative"]),
        mean(method_scores["greedy"]),
        mean(method_scores["aggressive"]),
    ]
    ax.plot(x, means, color="#111111", linewidth=1.1, zorder=4)
    ax.scatter(x, means, marker="D", s=24, color="#111111", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Origin", "Conservative\nagent", "Greedy\ncontrol", "Aggressive\nagent"])
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(1.84, 2.09)
    ax.set_ylabel("Composite score $S$")
    ax.set_title("Matched-start score landscape", fontsize=7, fontweight="bold")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.55, zorder=0)
    panel_label(ax, "a")


def plot_score_gains(ax: plt.Axes, summary: list[dict]) -> None:
    lookup = {(row["method"], row["source_pool"]): row for row in summary}
    values = {
        method: [lookup[(method, pool)]["delta_score_mean"] for pool in SUMMARY_POOLS]
        for method in METHOD_ORDER
    }
    errors = {
        method: [lookup[(method, pool)]["delta_score_sem"] for pool in SUMMARY_POOLS]
        for method in METHOD_ORDER
    }
    grouped_bar_panel(
        ax,
        [POOL_LABELS[pool] for pool in SUMMARY_POOLS],
        values,
        errors,
        "$\\Delta S$",
    )
    ax.set_title("Composite-score gains", fontsize=7, fontweight="bold")
    ax.set_ylim(-0.035, 0.145)
    ax.tick_params(axis="x", labelrotation=25)
    panel_label(ax, "b")


def plot_property_changes(ax: plt.Axes, summary: list[dict]) -> None:
    lookup = {(row["method"], row["source_pool"]): row for row in summary}
    metrics = [
        ("delta_solubility", "$\\Delta p_{\\mathrm{sol}}$"),
        ("delta_non_fouling", "$\\Delta p_{\\mathrm{nf}}$"),
        ("delta_antihemolysis", "$\\Delta p_{\\mathrm{antihem}}$"),
    ]
    values = {
        method: [lookup[(method, "overall")][f"{metric}_mean"] for metric, _ in metrics]
        for method in METHOD_ORDER
    }
    errors = {
        method: [lookup[(method, "overall")][f"{metric}_sem"] for metric, _ in metrics]
        for method in METHOD_ORDER
    }
    grouped_bar_panel(
        ax,
        [label for _, label in metrics],
        values,
        errors,
        "Mean desired-direction change",
    )
    ax.set_title("Property-direction changes", fontsize=7, fontweight="bold")
    ax.set_ylim(-0.025, 0.065)
    panel_label(ax, "c")


def plot_length_changes(ax: plt.Axes, summary: list[dict]) -> None:
    lookup = {(row["method"], row["source_pool"]): row for row in summary}
    counts = [lookup[(method, "overall")]["length_changed_count"] for method in METHOD_ORDER]
    mean_lengths = [lookup[(method, "overall")]["delta_length_mean"] for method in METHOD_ORDER]
    xs = list(range(len(METHOD_ORDER)))
    bars = ax.bar(
        xs,
        counts,
        width=0.54,
        color=[METHOD_BY_KEY[method].color for method in METHOD_ORDER],
        edgecolor="#333333",
        linewidth=0.35,
        zorder=3,
    )

    for bar, count, delta_length in zip(bars, counts, mean_lengths):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.8,
            f"{count}/24",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(0.7, count * 0.5),
            f"mean {delta_length:+.2f} aa",
            ha="center",
            va="center",
            fontsize=5.8,
            color="#222222",
            rotation=90 if count < 3 else 0,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(["Conservative\nagent", "Greedy\ncontrol", "Aggressive\nagent"])
    ax.set_ylim(0, 25)
    ax.set_ylabel("Starts with length change")
    ax.set_title("Edit-policy breadth", fontsize=7, fontweight="bold")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.55, zorder=0)
    panel_label(ax, "d")


def build_figure(paths: Paths, records: list[dict], summary: list[dict]) -> None:
    fig = plt.figure(figsize=(183 / 25.4, 124 / 25.4), dpi=300)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.35, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.36,
        hspace=0.52,
    )
    plot_score_trajectories(fig.add_subplot(grid[0, 0]), records)
    plot_score_gains(fig.add_subplot(grid[0, 1]), summary)
    plot_property_changes(fig.add_subplot(grid[1, 0]), summary)
    plot_length_changes(fig.add_subplot(grid[1, 1]), summary)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            color=method.color,
            markeredgecolor="#333333",
            markeredgewidth=0.35,
            markersize=5,
            label=method.label,
        )
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.02),
        ncol=3,
        handletextpad=0.45,
        columnspacing=1.4,
    )
    save_publication_figure(fig, paths.out_dir / "aggressive_refinement_comparison")
    plt.close(fig)


def summary_lookup(summary: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["method"], row["source_pool"]): row for row in summary}


def write_table_tex(paths: Paths, summary: list[dict]) -> None:
    lookup = summary_lookup(summary)
    lines = [
        "\\begin{table}[H]",
        "    \\centering",
        "    \\small",
        "    \\caption{Aggressive Pepti-Agent refinement on the same 24 representative starting peptides. Positive values indicate improvement in the desired direction; $p_{\\mathrm{antihem}}=1-p_{\\mathrm{hem}}$ reports the desired reduction in predicted hemolysis.}",
        "    \\label{tab:aggressive_refinement}",
        "    \\begin{tabular}{lrrrrrrr}",
        "        \\toprule",
        "        Source & $n$ & $S_{\\mathrm{orig}}$ & $S_{\\mathrm{aggr}}$ & $\\Delta S$ & $\\Delta p_{\\mathrm{sol}}$ & $\\Delta p_{\\mathrm{nf}}$ & $\\Delta p_{\\mathrm{antihem}}$ \\\\",
        "        \\midrule",
    ]
    for pool in SUMMARY_POOLS:
        row = lookup[("aggressive", pool)]
        lines.append(
            "        "
            f"{POOL_LABELS[pool]} & {row['n']} & "
            f"{row['original_score_mean']:.3f} & {row['final_score_mean']:.3f} & "
            f"{row['delta_score_mean']:.3f} & {row['delta_solubility_mean']:.3f} & "
            f"{row['delta_non_fouling_mean']:.3f} & {row['delta_antihemolysis_mean']:.3f} \\\\"
        )
    lines.extend(["        \\bottomrule", "    \\end{tabular}", "\\end{table}", ""])
    (paths.out_dir / "aggressive_refinement_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_caption(paths: Paths, summary: list[dict], head_to_head: list[dict]) -> None:
    lookup = summary_lookup(summary)
    aggressive = lookup[("aggressive", "overall")]
    gt_conservative = sum(row["aggressive_gt_conservative"] for row in head_to_head)
    gt_greedy = sum(row["aggressive_gt_greedy"] for row in head_to_head)

    caption = (
        "Figure X. Aggressive refinement improves matched-start composite scores but "
        "expands edit depth. (a) Composite score for each matched start under the origin, "
        "conservative Pepti-Agent, one-step non-agent greedy, and aggressive Pepti-Agent "
        "outputs. Grey lines connect alternative outputs from the same start, and black "
        "diamonds show method means. (b) Mean composite-score change by source pool and "
        "overall. (c) Mean desired-direction property changes for solubility, non-fouling, "
        "and antihemolysis. Error bars in b and c indicate standard error of the mean. "
        "(d) Number of starts whose final peptide length changed under each method. "
        f"Aggressive refinement improved $S$ in {aggressive['score_improved_count']}/24 "
        f"starts, exceeded the conservative agent in {gt_conservative}/24 starts, and "
        f"exceeded the one-step greedy output in {gt_greedy}/24 starts, but changed "
        f"length in {aggressive['length_changed_count']}/24 cases.\n"
    )
    (paths.out_dir / "aggressive_refinement_caption.txt").write_text(
        caption, encoding="utf-8"
    )


def write_qa(paths: Paths, summary: list[dict], head_to_head: list[dict]) -> None:
    lookup = summary_lookup(summary)
    aggressive = lookup[("aggressive", "overall")]
    conservative = lookup[("conservative", "overall")]
    greedy = lookup[("greedy", "overall")]
    gt_conservative = sum(row["aggressive_gt_conservative"] for row in head_to_head)
    gt_greedy = sum(row["aggressive_gt_greedy"] for row in head_to_head)
    mean_aggressive_minus_conservative = mean(
        row["aggressive_minus_conservative"] for row in head_to_head
    )
    mean_aggressive_minus_greedy = mean(
        row["aggressive_minus_greedy"] for row in head_to_head
    )

    tiff_path = paths.out_dir / "aggressive_refinement_comparison.tiff"
    png_path = paths.out_dir / "aggressive_refinement_comparison.png"
    with Image.open(tiff_path) as image:
        tiff_size = image.size
        stat = ImageStat.Stat(image.convert("L"))
        tiff_mean = stat.mean[0]
        tiff_std = stat.stddev[0]
    with Image.open(png_path) as image:
        png_size = image.size
    svg_text = (paths.out_dir / "aggressive_refinement_comparison.svg").read_text(
        encoding="utf-8"
    )
    editable_text = "<text" in svg_text

    qa = f"""# Aggressive Refinement Figure QA

Core conclusion: aggressive refinement improves matched-start composite scores relative to conservative agent refinement and usually relative to one-step greedy search, but the gain is tied to a broader edit policy that often changes sequence length.

Archetype: quantitative grid.

Backend: Python/matplotlib only.

Evidence chain:
- Panel a: matched-start score landscape across origin, conservative agent, greedy control, and aggressive agent.
- Panel b: composite-score gains by source pool and overall.
- Panel c: desired-direction property changes that explain the score gains.
- Panel d: edit-policy caveat through length-change counts.

Key checks:
- Aggressive mean S: {aggressive['final_score_mean']:.6f}; origin mean S: {aggressive['original_score_mean']:.6f}; mean gain: {aggressive['delta_score_mean']:.6f}.
- Conservative mean gain: {conservative['delta_score_mean']:.6f}; greedy mean gain: {greedy['delta_score_mean']:.6f}.
- Aggressive vs conservative: {gt_conservative}/24 starts; mean advantage {mean_aggressive_minus_conservative:.6f}.
- Aggressive vs greedy: {gt_greedy}/24 starts; mean advantage {mean_aggressive_minus_greedy:.6f}.
- Length changed under aggressive refinement: {aggressive['length_changed_count']}/24 starts; mean length change {aggressive['delta_length_mean']:.6f} aa.
- Final threshold failures under aggressive refinement: {aggressive['threshold_fail_final_count']}/24.
- SVG editable text check: {'PASS' if editable_text else 'FAIL'}.
- TIFF size: {tiff_size[0]} x {tiff_size[1]} px at 600 dpi export target.
- PNG preview size: {png_size[0]} x {png_size[1]} px.
- Nonblank raster check: mean grayscale {tiff_mean:.2f}, stddev {tiff_std:.2f}.

Statistics notes:
- n = 24 matched starts overall; n = 8 per source pool.
- Error bars in panels b and c are standard error of the mean across matched starts.
- Metrics are model-internal predictor outputs; no wet-lab validation is represented.

Export bundle:
- aggressive_refinement_comparison.svg
- aggressive_refinement_comparison.pdf
- aggressive_refinement_comparison.tiff
- aggressive_refinement_comparison.png
- aggressive_comparison_matched_records.csv
- aggressive_summary_stats.csv
- aggressive_head_to_head.csv
- aggressive_refinement_caption.txt
- aggressive_refinement_table.tex
- make_aggressive_refinement_figure.py
"""
    (paths.out_dir / "aggressive_refinement_qa.md").write_text(qa, encoding="utf-8")


def write_manuscript_artifacts(paths: Paths, summary: list[dict], head_to_head: list[dict]) -> None:
    write_table_tex(paths, summary)
    write_caption(paths, summary, head_to_head)
    write_qa(paths, summary, head_to_head)


def main() -> None:
    paths = project_paths()
    records = load_all_records(paths)
    summary = summarize_records(records)
    head_to_head = head_to_head_records(records)

    export_source_data(paths, records, summary, head_to_head)
    build_figure(paths, records, summary)
    write_manuscript_artifacts(paths, summary, head_to_head)


if __name__ == "__main__":
    main()
