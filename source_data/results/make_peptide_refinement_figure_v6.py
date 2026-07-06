"""v6 refinement figure: 3-panel main + 1-panel supplementary gate figure.

Restructure per Option 1:
- Panel a: forest plot of paired Cohen's d_z (95% CI) with Cohen's interpretation bands
- Panel b: paired per-peptide score change distribution with mean 95% CI and median
- Panel c: source-set heatmap with cleaned annotations
- Supplementary: gate sensitivity table at thresholds 0.5-0.7
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "result"
DELTA_COUNT_THRESHOLD = 0.005
GATE_SENSITIVITY_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7)
DZ_NEGLIGIBLE = 0.2
DZ_SMALL = 0.5
DZ_MEDIUM = 0.8

PALETTE = {
    "origin": "#747474",
    "refined": "#0F4D92",
    "better": "#2E9E44",
    "worse": "#B64342",
    "neutral": "#D0D0D0",
    "neutral_dark": "#7A7A7A",
    "dark": "#252525",
    "grid": "#D8D8D8",
    "amber": "#B07A1A",
    "near_band": "#EEEEEE",
    "band_negl": "#F5F5F5",
    "band_small": "#E2EBF3",
    "band_medium": "#C9DAE9",
}

PLOT_PROPERTIES = [
    {"key": "sol", "label": "Solubility", "short": "Sol", "gate": True, "desired_delta": "sol_desired_delta"},
    {"key": "nf", "label": "Non-fouling", "short": "NF", "gate": True, "desired_delta": "nf_desired_delta"},
    {
        "key": "antihem",
        "label": "Non-hemolytic score",
        "short": "1-Hem",
        "gate": True,
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


def load_base_module():
    script_path = ROOT / "make_peptide_refinement_figure.py"
    spec = importlib.util.spec_from_file_location("peptide_refinement_base", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_ci(values: np.ndarray, seed: int = 7, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[sample_idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).astype(float))


def bootstrap_dz_ci(deltas: np.ndarray, seed: int = 11, n_boot: int = 10000) -> tuple[float, float]:
    deltas = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    samples = deltas[sample_idx]
    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dz = np.where(sds > 0, means / sds, np.nan)
    return tuple(np.nanpercentile(dz, [2.5, 97.5]).astype(float))


def prepare_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    base = load_base_module()
    df = base.build_source_data(ROOT.parent / "origin", ROOT.parent / "refined")
    summary = add_inference_columns(df, base.make_property_summary(df))
    gates = base.make_gate_summary(df)

    score_rows = []
    delta_rows = []
    for prop in PLOT_PROPERTIES:
        key = prop["key"]
        for _, row in df.iterrows():
            origin = float(row[f"origin_{key}"])
            refined = float(row[f"refined_{key}"])
            desired_delta = float(row[prop["desired_delta"]])
            common = {
                "peptide_id": row["peptide_id"],
                "source_partition": row["source_partition"],
                "idx": int(row["idx"]),
                "property": prop["label"],
                "property_short": prop["short"],
                "has_gate": bool(prop["gate"]),
                "gate_threshold": 0.5 if prop["gate"] else "",
            }
            score_rows.append({**common, "stage": "Origin", "score": origin})
            score_rows.append({**common, "stage": "Refined", "score": refined})
            delta_rows.append(
                {
                    **common,
                    "origin_score": origin,
                    "refined_score": refined,
                    "desired_delta": desired_delta,
                    "change_class": "better" if desired_delta > 0 else ("unchanged" if desired_delta == 0 else "worse"),
                    "change_class_thresholded": (
                        "better"
                        if desired_delta > DELTA_COUNT_THRESHOLD
                        else ("worse" if desired_delta < -DELTA_COUNT_THRESHOLD else "near_zero")
                    ),
                }
            )

    score_long = pd.DataFrame(score_rows)
    delta_long = pd.DataFrame(delta_rows)

    gate_transition = make_gate_transition(df)
    gate_sensitivity = make_gate_sensitivity(df)
    subgroup_delta = make_subgroup_delta(df)
    return df, summary, gates, score_long, delta_long, gate_transition, gate_sensitivity, subgroup_delta


def add_inference_columns(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        if row["property"] == "Non-hemolytic score (1-Hem)":
            prop = next(prop for prop in PLOT_PROPERTIES if prop["key"] == "antihem")
        else:
            prop = next(prop for prop in PLOT_PROPERTIES if prop["label"] == row["property"])
        values = df[prop["desired_delta"]].to_numpy(dtype=float)
        sd = float(values.std(ddof=1))
        dz = float(values.mean() / sd) if sd > 0 else np.nan
        dz_ci_low, dz_ci_high = bootstrap_dz_ci(values, seed=11 + hash(prop["key"]) % 1000)
        try:
            wilcoxon = stats.wilcoxon(values, alternative="two-sided", zero_method="wilcox")
            p_two_sided = float(wilcoxon.pvalue)
        except ValueError:
            p_two_sided = np.nan
        updated = row.to_dict()
        updated["paired_effect_size_dz"] = dz
        updated["paired_effect_size_dz_ci95_low"] = dz_ci_low
        updated["paired_effect_size_dz_ci95_high"] = dz_ci_high
        updated["wilcoxon_p_two_sided"] = p_two_sided
        updated["wilcoxon_p_two_sided_bonferroni_4"] = min(p_two_sided * len(PLOT_PROPERTIES), 1.0)
        updated["count_threshold"] = DELTA_COUNT_THRESHOLD
        updated["threshold_better_count"] = int((values > DELTA_COUNT_THRESHOLD).sum())
        updated["threshold_near_zero_count"] = int((np.abs(values) <= DELTA_COUNT_THRESHOLD).sum())
        updated["threshold_worse_count"] = int((values < -DELTA_COUNT_THRESHOLD).sum())
        rows.append(updated)
    return pd.DataFrame(rows)


def make_gate_transition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gate_defs = [
        ("Sol", "origin_pass_sol", "refined_pass_sol"),
        ("NF", "origin_pass_nf", "refined_pass_nf"),
        ("1-Hem", "origin_pass_antihem", "refined_pass_antihem"),
        ("All", "origin_pass_all", "refined_pass_all"),
    ]
    for gate, origin_col, refined_col in gate_defs:
        origin_pass = df[origin_col].astype(bool)
        refined_pass = df[refined_col].astype(bool)
        rows.append(
            {
                "gate": gate,
                "origin_failed": int((~origin_pass).sum()),
                "refined_failed": int((~refined_pass).sum()),
                "rescued_fail_to_pass": int((~origin_pass & refined_pass).sum()),
                "new_fail_pass_to_fail": int((origin_pass & ~refined_pass).sum()),
                "stable_pass": int((origin_pass & refined_pass).sum()),
                "stable_fail": int((~origin_pass & ~refined_pass).sum()),
                "n": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


def make_gate_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in GATE_SENSITIVITY_THRESHOLDS:
        origin_pass = (
            (df["origin_sol"] >= threshold)
            & (df["origin_nf"] >= threshold)
            & (df["origin_antihem"] >= threshold)
        )
        refined_pass = (
            (df["refined_sol"] >= threshold)
            & (df["refined_nf"] >= threshold)
            & (df["refined_antihem"] >= threshold)
        )
        origin_count = int(origin_pass.sum())
        refined_count = int(refined_pass.sum())
        rows.append(
            {
                "threshold": threshold,
                "origin_pass": origin_count,
                "refined_pass": refined_count,
                "origin_pass_percent": origin_count / len(df) * 100.0,
                "refined_pass_percent": refined_count / len(df) * 100.0,
                "rescued_fail_to_pass": int((~origin_pass & refined_pass).sum()),
                "new_fail_pass_to_fail": int((origin_pass & ~refined_pass).sum()),
                "net_pass_delta": refined_count - origin_count,
                "n": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


def make_subgroup_delta(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for partition, sub in df.groupby("source_partition", sort=True):
        for prop in PLOT_PROPERTIES:
            values = sub[prop["desired_delta"]].to_numpy(dtype=float)
            rows.append(
                {
                    "source_partition": partition,
                    "property": prop["label"],
                    "property_short": prop["short"],
                    "mean_desired_delta": float(values.mean()),
                    "median_desired_delta": float(np.median(values)),
                    "improved_count": int((values > 0).sum()),
                    "worsened_count": int((values < 0).sum()),
                    "unchanged_count": int((values == 0).sum()),
                    "threshold_better_count": int((values > DELTA_COUNT_THRESHOLD).sum()),
                    "threshold_near_zero_count": int((np.abs(values) <= DELTA_COUNT_THRESHOLD).sum()),
                    "threshold_worse_count": int((values < -DELTA_COUNT_THRESHOLD).sum()),
                    "n": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.11, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="black",
    )


def color_for_effect(delta: float, dz: float) -> str:
    if not np.isfinite(dz) or abs(dz) < DZ_NEGLIGIBLE:
        return PALETTE["neutral_dark"]
    if delta > 0:
        return PALETTE["better"]
    return PALETTE["worse"]


def get_stat_row(summary: pd.DataFrame, prop: dict) -> pd.Series:
    stat = summary[summary["property"] == prop["label"]]
    if stat.empty and prop["key"] == "antihem":
        stat = summary[summary["property"] == "Non-hemolytic score (1-Hem)"]
    return stat.iloc[0]


def plot_effect_sizes(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Panel a: paired Cohen's d_z forest plot with Cohen interpretation bands."""
    positions = np.arange(len(PLOT_PROPERTIES))

    band_specs = [
        (-DZ_MEDIUM, -DZ_SMALL, PALETTE["band_medium"]),
        (-DZ_SMALL, -DZ_NEGLIGIBLE, PALETTE["band_small"]),
        (-DZ_NEGLIGIBLE, DZ_NEGLIGIBLE, PALETTE["band_negl"]),
        (DZ_NEGLIGIBLE, DZ_SMALL, PALETTE["band_small"]),
        (DZ_SMALL, DZ_MEDIUM, PALETTE["band_medium"]),
    ]
    for low, high, color in band_specs:
        ax.axvspan(low, high, color=color, alpha=0.85, zorder=0)

    ax.axvline(0, color=PALETTE["dark"], linewidth=0.8, zorder=1)

    x_annot = DZ_MEDIUM + 0.06
    for pos, prop in zip(positions, PLOT_PROPERTIES):
        row = get_stat_row(summary, prop)
        dz = float(row["paired_effect_size_dz"])
        dz_low = float(row["paired_effect_size_dz_ci95_low"])
        dz_high = float(row["paired_effect_size_dz_ci95_high"])
        delta = float(row["mean_desired_delta"])
        p_bonf = float(row["wilcoxon_p_two_sided_bonferroni_4"])
        is_trivial = (not np.isfinite(dz)) or abs(dz) < DZ_NEGLIGIBLE
        color = color_for_effect(delta, dz)

        ax.errorbar(
            dz,
            pos,
            xerr=[[dz - dz_low], [dz_high - dz]],
            fmt="none",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3.0,
            zorder=4,
        )
        ax.scatter(
            [dz],
            [pos],
            s=46 if not is_trivial else 28,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )

        sig = "*" if (np.isfinite(p_bonf) and p_bonf < 0.05 and not is_trivial) else ""
        ax.text(
            x_annot,
            pos,
            f"d_z {dz:+.2f}{sig}  Delta {delta:+.4f}",
            ha="left",
            va="center",
            fontsize=7.0,
            color=color if not is_trivial else PALETTE["neutral_dark"],
            fontweight="bold" if not is_trivial else "normal",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([prop["short"] for prop in PLOT_PROPERTIES])
    ax.set_ylim(len(PLOT_PROPERTIES) - 0.35, -0.55)
    ax.set_xlim(-DZ_MEDIUM - 0.05, DZ_MEDIUM + 0.55)
    ax.set_xticks([-0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8])
    ax.set_xlabel(
        "Paired Cohen's d_z (95% bootstrap CI; shaded bands: |d_z| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium)"
    )
    ax.set_title("Effect size of refinement on predicted properties", loc="left", pad=14)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)

    ax.text(
        0.0,
        1.04,
        "n = 300 paired peptides    *  P_bonf < 0.05 (paired Wilcoxon, 4 tests)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=PALETTE["neutral_dark"],
    )
    add_panel_label(ax, "a", x=-0.055, y=1.14)


def plot_delta(ax: plt.Axes, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Panel b: paired score-change distribution, cleaned-up."""
    positions = np.arange(len(PLOT_PROPERTIES))
    arrays = [df[prop["desired_delta"]].to_numpy(dtype=float) for prop in PLOT_PROPERTIES]
    max_abs = max(abs(np.concatenate(arrays)).max(), 0.01) * 1.10
    ax.set_xlim(-max_abs, max_abs)

    ax.axvspan(
        -DELTA_COUNT_THRESHOLD,
        DELTA_COUNT_THRESHOLD,
        color=PALETTE["near_band"],
        alpha=0.85,
        zorder=-1,
    )

    for pos, prop, values in zip(positions, PLOT_PROPERTIES, arrays):
        row = get_stat_row(summary, prop)
        mean = float(row["mean_desired_delta"])
        ci_low = float(row["mean_desired_delta_ci95_low"])
        ci_high = float(row["mean_desired_delta_ci95_high"])
        median = float(row["median_desired_delta"])
        dz = float(row["paired_effect_size_dz"])
        color = color_for_effect(mean, dz)

        violin = ax.violinplot(
            [values],
            positions=[pos],
            vert=False,
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(PALETTE["neutral"])
            body.set_edgecolor("none")
            body.set_alpha(0.45)

        rng = np.random.default_rng(200 + pos)
        y = rng.normal(pos, 0.05, size=values.size)
        ax.scatter(
            values,
            y,
            s=4.0,
            color=PALETTE["neutral_dark"],
            alpha=0.18,
            linewidths=0,
            rasterized=True,
        )

        ax.scatter(
            [median],
            [pos - 0.22],
            s=90,
            marker="|",
            color=PALETTE["dark"],
            linewidth=2.2,
            zorder=4,
        )
        ax.errorbar(
            [mean],
            [pos + 0.22],
            xerr=[[mean - ci_low], [ci_high - mean]],
            fmt="o",
            markersize=7.0,
            color=color,
            ecolor=color,
            elinewidth=1.6,
            capsize=3.0,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=5,
        )

    ax.axvline(0, color="#9E9E9E", linewidth=0.8, zorder=0)
    ax.set_yticks(positions)
    ax.set_yticklabels([prop["short"] for prop in PLOT_PROPERTIES])
    ax.invert_yaxis()
    ax.set_xlabel("Refined - origin score")
    ax.set_title("Paired score changes per peptide", loc="left", pad=6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.text(
        0.01,
        0.98,
        "mean +/- 95% CI    | median    shaded: |Delta| <= 0.005",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color=PALETTE["dark"],
    )
    add_panel_label(ax, "b", x=-0.12)


def plot_subgroup_heatmap(ax: plt.Axes, subgroup_delta: pd.DataFrame) -> None:
    """Panel c: source-set heatmap with kept boxes and a short subtitle."""
    row_order = ["sampled_hem", "sampled_nf", "sampled_sol"]
    row_labels = ["Hem set", "NF set", "Sol set"]
    col_order = [prop["short"] for prop in PLOT_PROPERTIES]

    matrix = np.zeros((len(row_order), len(col_order)))
    for i, partition in enumerate(row_order):
        for j, prop in enumerate(col_order):
            row = subgroup_delta[
                (subgroup_delta["source_partition"] == partition) & (subgroup_delta["property_short"] == prop)
            ].iloc[0]
            matrix[i, j] = float(row["mean_desired_delta"])

    vmax = 0.03
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Mean score change by source set (n = 100 per set)", loc="left", pad=6)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if abs(value) > vmax * 0.55 else PALETTE["dark"]
            ax.text(
                j,
                i,
                f"{value:+.4f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color=text_color,
            )

    # Self-target degradation: NF set on NF column (i=1, j=1) - solid box
    ax.add_patch(
        Rectangle(
            (0.5, 0.5),
            1,
            1,
            fill=False,
            edgecolor=PALETTE["dark"],
            linewidth=1.4,
            zorder=5,
        )
    )
    # Cross-property cost: Sol set on NF column (i=2, j=1) - dashed box
    ax.add_patch(
        Rectangle(
            (0.5, 1.5),
            1,
            1,
            fill=False,
            edgecolor=PALETTE["dark"],
            linewidth=1.4,
            linestyle=(0, (3, 2)),
            zorder=5,
        )
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=6.2, width=0.5, length=2)
    cbar.set_label("Mean score change", fontsize=6.5)

    ax.text(
        0.0,
        -0.22,
        "Solid box: self-degradation (NF set -> NF).   Dashed box: cross-property cost (Sol set -> NF).",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=PALETTE["dark"],
    )
    add_panel_label(ax, "c", x=-0.14)


def plot_gate_sensitivity(ax: plt.Axes, sensitivity: pd.DataFrame) -> None:
    """Supplementary: all-property gate sensitivity table."""
    ax.axis("off")
    ax.set_title("Gate sensitivity (all-property pass)", loc="left", pad=6)

    columns = ["Gate threshold", "Origin pass", "Refined pass", "Rescued", "Newly failed", "Net"]
    x_pos = [0.04, 0.30, 0.52, 0.70, 0.85, 0.96]
    n_rows = len(sensitivity)
    y_header = 0.92
    y_top = 0.78
    y_bottom = 0.18
    row_h = (y_top - y_bottom) / (n_rows - 1) if n_rows > 1 else 0.0

    for x, col in zip(x_pos, columns):
        ax.text(
            x,
            y_header,
            col,
            transform=ax.transAxes,
            ha="center" if x > 0.1 else "left",
            va="center",
            fontsize=7.8,
            fontweight="bold",
        )
    ax.plot(
        [0.02, 0.98],
        [y_header - 0.05, y_header - 0.05],
        transform=ax.transAxes,
        color=PALETTE["grid"],
        linewidth=0.8,
    )

    for idx, row in sensitivity.reset_index(drop=True).iterrows():
        y = y_top - idx * row_h
        net = int(row["net_pass_delta"])
        values = [
            f">= {row['threshold']:.2f}",
            f"{int(row['origin_pass'])}/{int(row['n'])}",
            f"{int(row['refined_pass'])}/{int(row['n'])}",
            int(row["rescued_fail_to_pass"]),
            int(row["new_fail_pass_to_fail"]),
            f"{net:+d}",
        ]
        for x, value, col in zip(x_pos, values, columns):
            color = PALETTE["dark"]
            weight = "normal"
            if col == "Rescued" and isinstance(value, int) and value > 0:
                color = PALETTE["better"]
                weight = "bold"
            if col == "Newly failed" and isinstance(value, int) and value > 0:
                color = PALETTE["worse"]
                weight = "bold"
            if col == "Net":
                color = PALETTE["better"] if net > 0 else (PALETTE["worse"] if net < 0 else PALETTE["dark"])
                weight = "bold" if net != 0 else "normal"
            ax.text(
                x,
                y,
                str(value),
                transform=ax.transAxes,
                ha="center" if x > 0.1 else "left",
                va="center",
                fontsize=7.8,
                color=color,
                fontweight=weight,
            )
    ax.text(
        0.04,
        0.06,
        "Shared threshold on Sol, NF, and 1-Hem. Net = Refined pass - Origin pass.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=PALETTE["dark"],
    )


def make_main_figure(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    subgroup_delta: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.4,
            "axes.linewidth": 0.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.2, 5.3), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.18],
        height_ratios=[1.0, 1.25],
        wspace=0.45,
        hspace=0.55,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    plot_effect_sizes(ax_a, summary)
    plot_delta(ax_b, df, summary)
    plot_subgroup_heatmap(ax_c, subgroup_delta)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.10)
    base = OUT_DIR / "peptide_property_refinement_v6"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_supp_gate_figure(gate_sensitivity: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.6,
            "axes.linewidth": 0.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(5.0, 2.4))
    ax = fig.add_subplot(111)
    plot_gate_sensitivity(ax, gate_sensitivity)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.05)
    base = OUT_DIR / "peptide_property_refinement_v6_supp_gate"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    gates: pd.DataFrame,
    score_long: pd.DataFrame,
    delta_long: pd.DataFrame,
    transition: pd.DataFrame,
    gate_sensitivity: pd.DataFrame,
    subgroup_delta: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "source_data_peptide_refinement_v6.csv", index=False)
    summary.to_csv(OUT_DIR / "score_summary_v6.csv", index=False)
    gates.to_csv(OUT_DIR / "quality_gate_summary_v6.csv", index=False)
    score_long.to_csv(OUT_DIR / "score_long_v6.csv", index=False)
    delta_long.to_csv(OUT_DIR / "score_delta_long_v6.csv", index=False)
    transition.to_csv(OUT_DIR / "gate_transition_v6.csv", index=False)
    gate_sensitivity.to_csv(OUT_DIR / "gate_sensitivity_v6.csv", index=False)
    subgroup_delta.to_csv(OUT_DIR / "subgroup_mean_delta_v6.csv", index=False)

    sol = summary[summary["property"] == "Solubility"].iloc[0]
    nf = summary[summary["property"] == "Non-fouling"].iloc[0]
    antihem = summary[summary["property"] == "Non-hemolytic score (1-Hem)"].iloc[0]
    total = summary[summary["property"] == "Total prediction score"].iloc[0]
    all_origin = gates[(gates["gate"] == "All three gates") & (gates["stage"] == "Origin")].iloc[0]
    all_refined = gates[(gates["gate"] == "All three gates") & (gates["stage"] == "Refined")].iloc[0]

    contract = """# Figure contract v6

Restructure (Option 1): separate score-space (main figure) from gate-space (supplement).

Main figure (3 panels):
- Panel a: paired Cohen's d_z forest plot with 95% bootstrap CI; Cohen's interpretation bands shown as background shading (negligible |d_z| < 0.2, small 0.2-0.5, medium 0.5-0.8). Right-side annotation per row reports d_z and mean Delta. A trailing * marks rows passing Bonferroni-corrected paired Wilcoxon (4 tests, alpha = 0.05). Rows with |d_z| < 0.2 are rendered neutral gray and unbolded to flag negligible effects regardless of sign.
- Panel b: paired per-peptide score-change distribution. Violins and individual points are rendered neutral gray to keep visual mass off noise; the median is shown as a black vertical tick; the mean and 95% bootstrap CI is shown as a colored point with horizontal whiskers. The near-zero band (|Delta| <= 0.005) is shaded.
- Panel c: source-set heatmap of mean score change (n = 100 per set). One value per cell. A solid box marks self-target degradation (NF set on NF, -0.0101); a dashed box marks the cross-property cost (Sol set on NF, -0.0268) that exceeds the NF set's own self-degradation. Inline subtitle names the boxes.

Supplementary figure:
- Gate sensitivity table at thresholds 0.50, 0.55, 0.60, 0.65, 0.70 with columns Origin pass, Refined pass, Rescued, Newly failed, Net.

Core conclusion (unchanged from v5): refinement produces medium-effect improvements on solubility and the non-hemolytic score (paired Cohen's d_z ~ +0.5 each), introduces a small-to-medium non-fouling trade-off (d_z ~ -0.4), and leaves the composite Total score essentially unchanged (|d_z| < 0.2, rendered in neutral gray). The NF cost concentrates on Sol-source peptides, exceeding the NF set's own self-degradation.

Score definitions:
- Sol and NF use the original model probabilities.
- 1-Hem = 1 - hemolysis probability (higher is better).
- Total = mean(Sol, NF, 1-Hem).
- Hard gates are Sol >= 0.5, NF >= 0.5, and 1-Hem >= 0.5. Total is not treated as a hard gate.

Statistical conventions:
- 95% CIs use 10,000 bootstrap resamples of paired values (panel a x-axis: paired Delta; panel b mean marker: paired Delta).
- Effect size is paired Cohen's d_z = mean(Delta) / SD(Delta); CI is bootstrapped over paired differences.
- Cohen's interpretation: |d_z| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >= 0.8 large.
- P values use paired Wilcoxon signed-rank with Bonferroni correction across the four scores.
- Thresholded counts use |Delta| > 0.005, a conservative noise band.
"""
    (OUT_DIR / "figure_contract_v6.md").write_text(contract, encoding="utf-8")

    caption = """**Figure X | Refinement raises predicted solubility and non-hemolytic score with a measurable non-fouling trade-off.** All scores are shown in the higher-is-better direction: solubility (Sol), non-fouling (NF), non-hemolytic score (1-Hem = 1 - hemolysis probability), and total prediction score (mean of Sol, NF, and 1-Hem). **a**, Paired Cohen's d_z (effect size of refined - origin) with 95% bootstrap CI for n = 300 paired peptides; background bands indicate Cohen's interpretation (negligible, small, medium). Right-hand labels report d_z and the mean score change (Delta); * indicates P_bonf < 0.05 from paired Wilcoxon signed-rank tests with Bonferroni correction across the four scores. **b**, Distribution of per-peptide score changes; gray points and violins show the empirical distribution, vertical ticks show the median, and colored points show the mean with 95% bootstrap CI (10,000 resamples of paired differences). The shaded band marks |Delta| <= 0.005 (the near-zero noise band used for thresholded counts). **c**, Mean score change by source set (n = 100 each). The solid box highlights the non-fouling generator's self-degradation on its own target property (NF set -> NF, Delta = -0.0101); the dashed box highlights the larger cross-property cost where solubility-driven refinement degrades NF (Sol set -> NF, Delta = -0.0268).

**Supplementary Figure Sx | Refinement sensitivity to the all-property gate threshold.** Counts of peptides passing the shared score threshold on Sol, NF, and 1-Hem simultaneously, before and after refinement (n = 300). Rescued: origin-fail to refined-pass. Newly failed: origin-pass to refined-fail. Net: refined pass - origin pass.
"""
    (OUT_DIR / "figure_caption_v6.md").write_text(caption, encoding="utf-8")

    notes = f"""# QA notes v6

Backend: Python {sys.version.split()[0]} / matplotlib {matplotlib.__version__}.

Structural change vs v5:
- Main figure restructured to 3 panels (a/b/c). Panel a replaced with a d_z forest plot (was: mean-score forest). Panel b retained with reduced visual noise (uniform gray points; mean and median visually separated). Panel c (heatmap) promoted to bottom-right at increased size.
- Gate sensitivity moved to a separate supplementary figure (peptide_property_refinement_v6_supp_gate.*).

Key numeric checks:
- Solubility: mean Delta {sol['mean_desired_delta']:+.6f} (95% CI {sol['mean_desired_delta_ci95_low']:+.6f} to {sol['mean_desired_delta_ci95_high']:+.6f}); d_z {sol['paired_effect_size_dz']:+.3f} (95% CI {sol['paired_effect_size_dz_ci95_low']:+.3f} to {sol['paired_effect_size_dz_ci95_high']:+.3f}); P_bonf {sol['wilcoxon_p_two_sided_bonferroni_4']:.3e}.
- Non-fouling: mean Delta {nf['mean_desired_delta']:+.6f} (95% CI {nf['mean_desired_delta_ci95_low']:+.6f} to {nf['mean_desired_delta_ci95_high']:+.6f}); d_z {nf['paired_effect_size_dz']:+.3f} (95% CI {nf['paired_effect_size_dz_ci95_low']:+.3f} to {nf['paired_effect_size_dz_ci95_high']:+.3f}); P_bonf {nf['wilcoxon_p_two_sided_bonferroni_4']:.3e}.
- 1-Hem: mean Delta {antihem['mean_desired_delta']:+.6f} (95% CI {antihem['mean_desired_delta_ci95_low']:+.6f} to {antihem['mean_desired_delta_ci95_high']:+.6f}); d_z {antihem['paired_effect_size_dz']:+.3f} (95% CI {antihem['paired_effect_size_dz_ci95_low']:+.3f} to {antihem['paired_effect_size_dz_ci95_high']:+.3f}); P_bonf {antihem['wilcoxon_p_two_sided_bonferroni_4']:.3e}.
- Total: mean Delta {total['mean_desired_delta']:+.6f} (95% CI {total['mean_desired_delta_ci95_low']:+.6f} to {total['mean_desired_delta_ci95_high']:+.6f}); d_z {total['paired_effect_size_dz']:+.3f} (95% CI {total['paired_effect_size_dz_ci95_low']:+.3f} to {total['paired_effect_size_dz_ci95_high']:+.3f}); P_bonf {total['wilcoxon_p_two_sided_bonferroni_4']:.3e}.
- All-gate pass rate (>= 0.5): origin {all_origin['passed_count']:.0f}/{all_origin['n']:.0f}; refined {all_refined['passed_count']:.0f}/{all_refined['n']:.0f}.
- Gate sensitivity thresholds: {GATE_SENSITIVITY_THRESHOLDS}.
- Panel a sig marker: * shown when |d_z| >= {DZ_NEGLIGIBLE} AND P_bonf < 0.05.
- Panel b threshold for better/near/worse counts: |Delta| > {DELTA_COUNT_THRESHOLD}.

Export bundle:
- peptide_property_refinement_v6.svg / .pdf / .png / .tiff
- peptide_property_refinement_v6_supp_gate.svg / .pdf / .png
- source_data_peptide_refinement_v6.csv
- score_summary_v6.csv
- quality_gate_summary_v6.csv
- score_long_v6.csv
- score_delta_long_v6.csv
- gate_transition_v6.csv
- gate_sensitivity_v6.csv
- subgroup_mean_delta_v6.csv
- figure_contract_v6.md
- figure_caption_v6.md
"""
    (OUT_DIR / "qa_notes_v6.md").write_text(notes, encoding="utf-8")


def main() -> None:
    df, summary, gates, score_long, delta_long, transition, gate_sensitivity, subgroup_delta = prepare_data()
    write_outputs(df, summary, gates, score_long, delta_long, transition, gate_sensitivity, subgroup_delta)
    make_main_figure(df, summary, subgroup_delta)
    make_supp_gate_figure(gate_sensitivity)
    print(f"Wrote v6 figure bundle to {OUT_DIR}")
    print(
        summary[
            [
                "property",
                "mean_desired_delta",
                "paired_effect_size_dz",
                "paired_effect_size_dz_ci95_low",
                "paired_effect_size_dz_ci95_high",
                "wilcoxon_p_two_sided_bonferroni_4",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
