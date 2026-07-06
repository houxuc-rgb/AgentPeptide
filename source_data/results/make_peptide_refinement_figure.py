from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


PALETTE = {
    "origin": "#767676",
    "refined": "#0F4D92",
    "better": "#2E9E44",
    "worse": "#B64342",
    "neutral": "#CFCECE",
    "dark": "#272727",
    "teal": "#42949E",
}

PROPERTIES = [
    {
        "key": "sol",
        "label": "Solubility",
        "short": "Sol",
        "gate": True,
        "threshold": 0.5,
        "pass_direction": "ge",
        "desired_delta": "sol_desired_delta",
    },
    {
        "key": "nf",
        "label": "Non-fouling",
        "short": "NF",
        "gate": True,
        "threshold": 0.5,
        "pass_direction": "ge",
        "desired_delta": "nf_desired_delta",
    },
    {
        "key": "antihem",
        "label": "Non-hemolytic score (1-Hem)",
        "short": "1-Hem",
        "gate": True,
        "gate_label": "1-Hemolysis >= 0.5",
        "threshold": 0.5,
        "pass_direction": "ge",
        "desired_delta": "antihem_desired_delta",
    },
    {
        "key": "composite_score",
        "label": "Total prediction score",
        "short": "Total",
        "gate": False,
        "desired_delta": "composite_delta",
    },
]


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path.name} is not a JSON list")
    return data


def pass_gate(values: pd.Series, direction: str, threshold: float) -> pd.Series:
    if direction == "ge":
        return values >= threshold
    if direction == "lt":
        return values < threshold
    raise ValueError(f"Unsupported direction: {direction}")


def bootstrap_ci(values: np.ndarray, seed: int = 7, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[sample_idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def build_source_data(origin_dir: Path, refined_dir: Path) -> pd.DataFrame:
    rows = []
    origin_files = sorted(origin_dir.glob("*.json"))
    refined_names = {path.name for path in refined_dir.glob("*.json")}
    if not origin_files:
        raise FileNotFoundError("No JSON files found in origin directory")

    for origin_path in origin_files:
        refined_path = refined_dir / origin_path.name
        if origin_path.name not in refined_names:
            raise FileNotFoundError(f"Missing refined match for {origin_path.name}")

        origin_items = {int(item["idx"]): item for item in load_json(origin_path)}
        refined_items = {int(item["idx"]): item for item in load_json(refined_path)}
        if set(origin_items) != set(refined_items):
            raise ValueError(f"Index mismatch for {origin_path.name}")

        partition = origin_path.stem
        for idx in sorted(origin_items):
            before = origin_items[idx]
            after = refined_items[idx]
            row = {
                "peptide_id": f"{partition}_{idx:03d}",
                "source_partition": partition,
                "idx": idx,
                "origin_sequence": before["sequence"],
                "refined_sequence": after["sequence"],
                "origin_length": int(before["length"]),
                "refined_length": int(after["length"]),
            }
            for prop in ("sol", "nf", "hem"):
                row[f"origin_{prop}"] = float(before[prop])
                row[f"refined_{prop}"] = float(after[prop])
                row[f"{prop}_delta"] = float(after[prop]) - float(before[prop])

            row["origin_antihem"] = 1.0 - row["origin_hem"]
            row["refined_antihem"] = 1.0 - row["refined_hem"]
            row["antihem_delta"] = row["refined_antihem"] - row["origin_antihem"]
            row["sol_desired_delta"] = row["sol_delta"]
            row["nf_desired_delta"] = row["nf_delta"]
            row["hem_desired_delta"] = -row["hem_delta"]
            row["antihem_desired_delta"] = row["antihem_delta"]
            row["origin_composite_score"] = (
                row["origin_sol"] + row["origin_nf"] + row["origin_antihem"]
            ) / 3.0
            row["refined_composite_score"] = (
                row["refined_sol"] + row["refined_nf"] + row["refined_antihem"]
            ) / 3.0
            row["composite_delta"] = row["refined_composite_score"] - row["origin_composite_score"]
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) != 300:
        raise ValueError(f"Expected 300 paired peptides, found {len(df)}")

    gate_properties = [prop for prop in PROPERTIES if prop.get("gate", False)]
    for prop in gate_properties:
        key = prop["key"]
        df[f"origin_pass_{key}"] = pass_gate(
            df[f"origin_{key}"], prop["pass_direction"], prop["threshold"]
        )
        df[f"refined_pass_{key}"] = pass_gate(
            df[f"refined_{key}"], prop["pass_direction"], prop["threshold"]
        )
    df["origin_pass_hem"] = df["origin_pass_antihem"]
    df["refined_pass_hem"] = df["refined_pass_antihem"]
    df["origin_pass_all"] = df[[f"origin_pass_{p['key']}" for p in gate_properties]].all(axis=1)
    df["refined_pass_all"] = df[[f"refined_pass_{p['key']}" for p in gate_properties]].all(axis=1)
    return df


def make_property_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prop in PROPERTIES:
        key = prop["key"]
        desired = df[prop["desired_delta"]].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(desired)
        improved = int((desired > 0).sum())
        unchanged = int((desired == 0).sum())
        worsened = int((desired < 0).sum())
        try:
            test = stats.wilcoxon(desired, alternative="greater", zero_method="wilcox")
            p_value = float(test.pvalue)
            statistic = float(test.statistic)
        except ValueError:
            p_value = np.nan
            statistic = np.nan
        rows.append(
            {
                "property": prop["label"],
                "n": int(len(df)),
                "origin_mean": float(df[f"origin_{key}"].mean()),
                "refined_mean": float(df[f"refined_{key}"].mean()),
                "origin_median": float(df[f"origin_{key}"].median()),
                "refined_median": float(df[f"refined_{key}"].median()),
                "mean_desired_delta": float(desired.mean()),
                "median_desired_delta": float(np.median(desired)),
                "mean_desired_delta_ci95_low": ci_low,
                "mean_desired_delta_ci95_high": ci_high,
                "improved_count": improved,
                "unchanged_count": unchanged,
                "worsened_count": worsened,
                "improved_percent": improved / len(df) * 100.0,
                "wilcoxon_statistic_greater_than_zero": statistic,
                "wilcoxon_p_greater_than_zero": p_value,
            }
        )
    return pd.DataFrame(rows)


def make_gate_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gate_properties = [prop for prop in PROPERTIES if prop.get("gate", False)]
    gates = [
        (
            prop.get("gate_label", f"{prop['label']} >= {prop['threshold']}"),
            f"origin_pass_{prop['key']}",
            f"refined_pass_{prop['key']}",
        )
        for prop in gate_properties
    ]
    gates.append(("All three gates", "origin_pass_all", "refined_pass_all"))
    for label, origin_col, refined_col in gates:
        for stage, col in [("Origin", origin_col), ("Refined", refined_col)]:
            passed = int(df[col].sum())
            rows.append(
                {
                    "gate": label,
                    "stage": stage,
                    "passed_count": passed,
                    "failed_count": int(len(df) - passed),
                    "pass_percent": passed / len(df) * 100.0,
                    "n": int(len(df)),
                }
            )
    return pd.DataFrame(rows)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="black",
    )


def plot_property_scores(ax: plt.Axes, df: pd.DataFrame) -> None:
    positions = np.arange(len(PROPERTIES))
    width = 0.26
    rng = np.random.default_rng(11)
    for pos, prop in zip(positions, PROPERTIES):
        key = prop["key"]
        origin = df[f"origin_{key}"].to_numpy(dtype=float)
        refined = df[f"refined_{key}"].to_numpy(dtype=float)
        for offset, values, color, label in [
            (-width / 2, origin, PALETTE["origin"], "Origin"),
            (width / 2, refined, PALETTE["refined"], "Refined"),
        ]:
            x = rng.normal(pos + offset, 0.025, size=values.size)
            ax.scatter(x, values, s=7, color=color, alpha=0.18, linewidths=0, rasterized=True)
            mean = float(values.mean())
            low, high = bootstrap_ci(values, seed=50 + pos + int(offset > 0))
            ax.errorbar(
                pos + offset,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                markersize=4,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2,
                label=label if pos == 0 else None,
                zorder=4,
            )
    ax.hlines(0.5, -0.42, 2.42, color="#A0A0A0", linewidth=0.8, linestyle="--")
    ax.text(
        2.42,
        0.505,
        "model gate",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=PALETTE["dark"],
    )
    ax.set_xlim(-0.48, len(PROPERTIES) - 0.52)
    ax.set_ylim(0.32, 0.77)
    ax.set_xticks(positions)
    ax.set_xticklabels([prop["short"] for prop in PROPERTIES])
    ax.set_ylabel("Predicted score")
    ax.set_title("Property scores before and after refinement", loc="left", pad=6)
    ax.legend(loc="lower left", fontsize=5.8, handlelength=1.0, borderpad=0.2)
    ax.text(
        0.98,
        0.07,
        "Higher is better for all scores;\nHem shown as 1-Hem",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=PALETTE["dark"],
    )
    add_panel_label(ax, "a", x=-0.18)


def plot_desired_delta(ax: plt.Axes, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    arrays = [df[prop["desired_delta"]].to_numpy(dtype=float) for prop in PROPERTIES]
    positions = np.arange(len(PROPERTIES))
    max_abs = max(abs(np.concatenate(arrays)).max(), 0.01) * 1.1
    ax.set_xlim(-max_abs, max_abs)
    for pos, values, prop in zip(positions, arrays, PROPERTIES):
        violin = ax.violinplot(
            [values],
            positions=[pos],
            vert=False,
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        median = float(np.median(values))
        color = PALETTE["better"] if median >= 0 else PALETTE["worse"]
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.28)

        rng = np.random.default_rng(100 + pos)
        y = rng.normal(pos, 0.045, size=values.size)
        ax.scatter(values, y, s=5, c=color, alpha=0.28, linewidths=0, rasterized=True)
        q1, q3 = np.percentile(values, [25, 75])
        ax.plot([q1, q3], [pos, pos], color=PALETTE["dark"], linewidth=1.2, solid_capstyle="round")
        ax.scatter([median], [pos], s=18, color=PALETTE["dark"], zorder=4)
        improved = int((values > 0).sum())
        ax.text(
            max_abs * 0.96,
            pos,
            f"{improved}/{len(values)} better",
            va="center",
            ha="right",
            fontsize=6,
            color=PALETTE["dark"],
        )

    ax.axvline(0, color="#A0A0A0", linewidth=0.8, zorder=0)
    ax.set_yticks(positions)
    ax.set_yticklabels([prop["label"] for prop in PROPERTIES])
    ax.invert_yaxis()
    ax.set_xlabel("Desired change in predicted score")
    ax.set_title("Per-peptide change in the desired direction", loc="left", pad=6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    add_panel_label(ax, "b")


def plot_gate_summary(ax: plt.Axes, gate_summary: pd.DataFrame) -> None:
    gate_properties = [prop for prop in PROPERTIES if prop.get("gate", False)]
    gates = [prop.get("gate_label", f"{prop['label']} >= 0.5") for prop in gate_properties] + [
        "All three gates"
    ]
    x = np.arange(len(gates))
    width = 0.36
    origin = [
        float(gate_summary[(gate_summary["gate"] == gate) & (gate_summary["stage"] == "Origin")]["failed_count"].iloc[0])
        for gate in gates
    ]
    refined = [
        float(gate_summary[(gate_summary["gate"] == gate) & (gate_summary["stage"] == "Refined")]["failed_count"].iloc[0])
        for gate in gates
    ]
    ax.bar(x - width / 2, origin, width=width, color=PALETTE["origin"], label="Origin")
    ax.bar(x + width / 2, refined, width=width, color=PALETTE["refined"], label="Refined")
    for x_pos, value in zip(x - width / 2, origin):
        ax.text(x_pos, value + 0.08, f"{value:.0f}", ha="center", va="bottom", fontsize=5.8)
    for x_pos, value in zip(x + width / 2, refined):
        ax.text(x_pos, value + 0.08, f"{value:.0f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_ylim(0, max(origin + refined + [1]) + 0.7)
    ax.set_ylabel("Failed peptides (n)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Sol", "NF", "1-Hem", "All"])
    ax.set_title("Property gate failures", loc="left", pad=6)
    ax.legend(loc="upper left", fontsize=5.8, handlelength=1.0, borderpad=0.2)
    add_panel_label(ax, "c", x=-0.2)


def make_figure(df: pd.DataFrame, property_summary: pd.DataFrame, gate_summary: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 6.5,
            "axes.linewidth": 0.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
        }
    )
    fig = plt.figure(figsize=(7.2, 3.85), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.18, 0.82],
        height_ratios=[1.0, 1.0],
        wspace=0.52,
        hspace=0.62,
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[:, 1])
    ax_c = fig.add_subplot(grid[:, 2])

    plot_property_scores(ax_a, df)
    plot_desired_delta(ax_b, df, property_summary)
    plot_gate_summary(ax_c, gate_summary)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.89, bottom=0.16)

    base = out_dir / "peptide_property_refinement"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_contract(out_dir: Path) -> None:
    text = """# Figure contract

Core conclusion: pooled peptide refinement increases solubility and the non-hemolytic score (1 - hemolysis probability), removing predicted hemolysis outliers, but it creates a measurable non-fouling trade-off; the total prediction score is reported explicitly to show the net effect of these opposing changes.

Archetype: quantitative grid.

Evidence chain:
- Panel a: predicted property scores before and after agent refinement, with hemolysis shown as 1 - hemolysis probability so all scores are higher-is-better; total prediction score is the mean of Sol, NF, and 1-Hem.
- Panel b: per-peptide score changes for all 300 paired candidates, including total prediction score.
- Panel c: model-gate failures before and after refinement.

Journal/export contract:
- Final size: double-column width, 183 mm by approximately 98 mm.
- Export formats: editable SVG, editable-text PDF, 600 dpi TIFF, and PNG preview.
- Source data: CSV tables in this folder.
- Statistics: paired summaries across 300 peptides; Wilcoxon signed-rank tests use the desired-change direction for each property.
- Image integrity: no microscopy or photographic image processing; all panels are generated directly from source JSON values.
"""
    (out_dir / "figure_contract.md").write_text(text, encoding="utf-8")


def write_qa_notes(out_dir: Path, df: pd.DataFrame, property_summary: pd.DataFrame, gate_summary: pd.DataFrame) -> None:
    sol = property_summary[property_summary["property"] == "Solubility"].iloc[0]
    nf = property_summary[property_summary["property"] == "Non-fouling"].iloc[0]
    antihem = property_summary[property_summary["property"] == "Non-hemolytic score (1-Hem)"].iloc[0]
    total = property_summary[property_summary["property"] == "Total prediction score"].iloc[0]
    all_origin = gate_summary[(gate_summary["gate"] == "All three gates") & (gate_summary["stage"] == "Origin")].iloc[0]
    all_refined = gate_summary[(gate_summary["gate"] == "All three gates") & (gate_summary["stage"] == "Refined")].iloc[0]
    text = f"""# QA notes

Backend: Python {sys.version.split()[0]} / matplotlib {matplotlib.__version__} with editable SVG text enabled.

Input handling:
- Paired records analysed: {len(df)}
- JSON partitions were used only to pair origin/refined records; plotted analyses pool all peptides.
- Thresholds: solubility >= 0.5, non-fouling >= 0.5, non-hemolytic score (1 - hemolysis probability) >= 0.5.

Key numeric checks:
- Solubility mean desired change: {sol['mean_desired_delta']:.6f} (95% bootstrap CI {sol['mean_desired_delta_ci95_low']:.6f} to {sol['mean_desired_delta_ci95_high']:.6f})
- Non-fouling mean desired change: {nf['mean_desired_delta']:.6f} (95% bootstrap CI {nf['mean_desired_delta_ci95_low']:.6f} to {nf['mean_desired_delta_ci95_high']:.6f})
- Non-hemolytic score mean desired change: {antihem['mean_desired_delta']:.6f} (95% bootstrap CI {antihem['mean_desired_delta_ci95_low']:.6f} to {antihem['mean_desired_delta_ci95_high']:.6f})
- Total prediction score mean change: {total['mean_desired_delta']:.6f} (95% bootstrap CI {total['mean_desired_delta_ci95_low']:.6f} to {total['mean_desired_delta_ci95_high']:.6f})
- All-gate pass rate: origin {all_origin['passed_count']:.0f}/{all_origin['n']:.0f}; refined {all_refined['passed_count']:.0f}/{all_refined['n']:.0f}.

Export bundle:
- peptide_property_refinement.svg
- peptide_property_refinement.pdf
- peptide_property_refinement.tiff
- peptide_property_refinement.png
- source_data_peptide_refinement.csv
- summary_statistics.csv
- quality_gate_summary.csv
- figure_contract.md
- qa_notes.md
"""
    (out_dir / "qa_notes.md").write_text(text, encoding="utf-8")


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    root = out_dir.parent
    origin_dir = root / "origin"
    refined_dir = root / "refined"

    df = build_source_data(origin_dir, refined_dir)
    property_summary = make_property_summary(df)
    gate_summary = make_gate_summary(df)

    df.to_csv(out_dir / "source_data_peptide_refinement.csv", index=False)
    property_summary.to_csv(out_dir / "summary_statistics.csv", index=False)
    gate_summary.to_csv(out_dir / "quality_gate_summary.csv", index=False)

    make_figure(df, property_summary, gate_summary, out_dir)
    write_contract(out_dir)
    write_qa_notes(out_dir, df, property_summary, gate_summary)


if __name__ == "__main__":
    main()
