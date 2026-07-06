from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats


ROOT = Path(__file__).resolve().parent
FIG2_DIR = ROOT
FIG3_DIR = ROOT
OUT_DIR = ROOT / "review_revisions"

METHOD_COLORS_F2 = {
    "Origin": "#767676",
    "Agent": "#4B4A7E",
    "Exhaustive": "#4A4A4A",
    "Gap": "#B0B0B0",
}
METHOD_COLORS_F3 = {
    "conservative": "#7884B4",
    "greedy": "#767676",
    "aggressive": "#C76F82",
    "aggressive_equal_length": "#D8A0AD",
}
POOL_COLORS = {"sol": "#5B7FCA", "nf": "#2EAD9B", "hem": "#D9544D"}
POOL_LABELS = {"sol": "Sol set", "nf": "NF set", "hem": "1-Hem set", "overall": "All"}
PROPERTY_LABELS = {
    "sol": "Sol",
    "nf": "NF",
    "antihem": "1-Hem",
    "score": "S",
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.2,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def bootstrap_ci(values: np.ndarray, seed: int = 7, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[sample].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).astype(float))


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, fontweight="bold")


def paired_effect(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1)
    return float(values.mean() / sd) if sd > 0 else np.nan


def hamming_or_levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_stats(rows: pd.DataFrame) -> pd.DataFrame:
    diff = rows["nonagent_minus_agent_S"].to_numpy(float)
    sign_p = stats.binomtest(int((diff > 0).sum()), len(diff), 0.5, alternative="greater").pvalue
    wilcoxon_p = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue
    out = [
        {
            "comparison": "nonagent_minus_agent_S",
            "n": len(diff),
            "mean": float(diff.mean()),
            "median": float(np.median(diff)),
            "ci95_low": bootstrap_ci(diff, seed=120)[0],
            "ci95_high": bootstrap_ci(diff, seed=120)[1],
            "nonagent_higher_count": int((diff > 0).sum()),
            "sign_test_p_greater": float(sign_p),
            "wilcoxon_p_greater": float(wilcoxon_p),
            "paired_effect_size_dz": paired_effect(diff),
        }
    ]
    return pd.DataFrame(out)


def plot_fig2() -> None:
    rows = pd.read_csv(FIG2_DIR / "matched_agent_nonagent_source.csv")
    prop_summary = pd.read_csv(FIG2_DIR / "summary_by_method_property.csv")
    pool_summary = pd.read_csv(FIG2_DIR / "summary_by_pool.csv")
    stats_summary = fig2_stats(rows)

    sorted_rows = rows.sort_values("origin_S", ascending=True).reset_index(drop=True)
    x = np.arange(len(sorted_rows))

    fig = plt.figure(figsize=(7.2, 5.1), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.34, 1.0], height_ratios=[1.0, 0.92], wspace=0.34, hspace=0.55)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    for i, row in sorted_rows.iterrows():
        edge = POOL_COLORS[row["source_pool"]]
        ymin = min(row["origin_S"], row["agent_S"], row["nonagent_S"])
        ymax = max(row["origin_S"], row["agent_S"], row["nonagent_S"])
        ax_a.plot([i, i], [ymin, ymax], color="#D4D4D4", lw=0.7, zorder=1)
        ax_a.scatter(i, row["origin_S"], s=16, facecolor=METHOD_COLORS_F2["Origin"], edgecolor=edge, lw=0.9, alpha=0.78, zorder=3)
        ax_a.scatter(i, row["agent_S"], s=26, facecolor=METHOD_COLORS_F2["Agent"], edgecolor=edge, lw=0.9, zorder=4)
        ax_a.scatter(i, row["nonagent_S"], s=33, marker="^", facecolor=METHOD_COLORS_F2["Exhaustive"], edgecolor=edge, lw=0.9, zorder=5)

    for _, col, color in [
        ("Origin mean", "origin_S", METHOD_COLORS_F2["Origin"]),
        ("Agent mean", "agent_S", METHOD_COLORS_F2["Agent"]),
        ("Search mean", "nonagent_S", METHOD_COLORS_F2["Exhaustive"]),
    ]:
        y = float(sorted_rows[col].mean())
        ax_a.axhline(y, color=color, lw=0.9, ls="--", alpha=0.9)

    ax_a.set_title("Fixed-length exhaustive search on matched starts", loc="left", pad=6)
    ax_a.set_ylabel("Composite score S")
    ax_a.set_xlabel("24 matched starts ordered by origin S")
    ax_a.set_xticks([])
    ax_a.grid(axis="y", color="#E5E5E5", lw=0.6)
    ax_a.text(
        0.02,
        0.96,
        "Balanced subset: 8 per source set,\n2 per source-set score quartile",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D0D0D0"),
    )
    ax_a.text(
        0.98,
        0.82,
        "Dashed lines: method means",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
    )
    add_panel_label(ax_a, "a")

    method_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS_F2["Origin"], markeredgecolor="#444444", label="Origin", markersize=4.6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS_F2["Agent"], markeredgecolor="#444444", label="Conservative agent", markersize=4.6),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=METHOD_COLORS_F2["Exhaustive"], markeredgecolor="#444444", label="Fixed-length exhaustive search", markersize=5.2),
    ]
    pool_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=POOL_COLORS[k], markeredgewidth=1.1, label=v, markersize=4.8)
        for k, v in POOL_LABELS.items()
        if k != "overall"
    ]
    leg1 = ax_a.legend(handles=method_handles, loc="lower right", fontsize=6.4, title="Fill", title_fontsize=6.3)
    ax_a.add_artist(leg1)
    ax_a.legend(handles=pool_handles, loc="lower left", fontsize=6.3, title="Edge", title_fontsize=6.3)

    headroom_specs = [
        ("Agent\n- origin", "agent_delta_S", METHOD_COLORS_F2["Agent"]),
        ("Search\n- origin", "nonagent_delta_S", METHOD_COLORS_F2["Exhaustive"]),
        ("Search\n- agent", "nonagent_minus_agent_S", METHOD_COLORS_F2["Gap"]),
    ]
    headroom_rows = []
    xpos = np.arange(len(headroom_specs))
    means, lows, highs = [], [], []
    for j, (label, col, color) in enumerate(headroom_specs):
        values = rows[col].to_numpy(float)
        low, high = bootstrap_ci(values, seed=180 + j)
        mean_value = float(values.mean())
        means.append(mean_value)
        lows.append(mean_value - low)
        highs.append(high - mean_value)
        headroom_rows.append(
            {
                "contrast": label.replace("\n", " "),
                "n": len(values),
                "mean_delta_S": mean_value,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    ax_b.bar(xpos, means, width=0.58, color=[spec[2] for spec in headroom_specs])
    ax_b.errorbar(xpos, means, yerr=[lows, highs], fmt="none", ecolor="#202020", lw=0.8, capsize=2)
    ax_b.axhline(0, color="#404040", lw=0.8)
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels([label for label, _, _ in headroom_specs], rotation=0, ha="center")
    ax_b.set_ylabel("Mean Delta S")
    ax_b.set_title("Local-search headroom (n = 24)", loc="left", pad=6)
    ax_b.text(
        0.02,
        1.00,
        "Fixed-length exhaustive search evaluates\nall one-substitution neighbors",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
    )
    ax_b.grid(axis="y", color="#E5E5E5", lw=0.6)
    add_panel_label(ax_b, "b")

    prop_order = [("sol", "Sol"), ("nf", "NF"), ("antihem", "1-Hem")]
    method_specs = [("Conservative agent", "agent", METHOD_COLORS_F2["Agent"]), ("Fixed-length exhaustive search", "nonagent", METHOD_COLORS_F2["Exhaustive"])]
    xpos = np.arange(len(prop_order))
    width = 0.34
    for j, (label, prefix, color) in enumerate(method_specs):
        means, lows, highs = [], [], []
        for prop, _ in prop_order:
            values = rows[f"{prefix}_delta_{prop}"].to_numpy(float)
            low, high = bootstrap_ci(values, seed=260 + j)
            mean_value = float(values.mean())
            means.append(mean_value)
            lows.append(mean_value - low)
            highs.append(high - mean_value)
        offset = (j - 0.5) * width
        ax_c.bar(xpos + offset, means, width=width, color=color, label=label)
        ax_c.errorbar(xpos + offset, means, yerr=[lows, highs], fmt="none", ecolor="#202020", lw=0.8, capsize=2)
    ax_c.axhline(0, color="#404040", lw=0.8)
    ax_c.set_xticks(xpos)
    ax_c.set_xticklabels([label for _, label in prop_order], rotation=0, ha="center")
    ax_c.set_ylabel("Mean property change")
    ax_c.set_title("Property-direction diagnosis", loc="left", pad=6)
    ax_c.legend(loc="upper left", bbox_to_anchor=(0.02, 1.00), borderaxespad=0.0, fontsize=6.2)
    ax_c.grid(axis="y", color="#E5E5E5", lw=0.6)
    add_panel_label(ax_c, "c")

    fig.suptitle("Fixed-length exhaustive search defines the local upper bound", y=0.995, fontsize=10.0)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.12)
    save_figure(fig, OUT_DIR / "agent_vs_nonagent_oracle_framing_v3")

    stats_summary.to_csv(OUT_DIR / "agent_vs_nonagent_head_to_head_stats_v3.csv", index=False)
    pd.DataFrame(headroom_rows).to_csv(OUT_DIR / "agent_vs_nonagent_headroom_stats_v3.csv", index=False)
    prop_summary.to_csv(OUT_DIR / "agent_vs_nonagent_original_property_summary_v3.csv", index=False)
    pool_summary.to_csv(OUT_DIR / "agent_vs_nonagent_original_pool_summary_v3.csv", index=False)

    caption = (
        "**Figure X | Fixed-length exhaustive search defines the local upper bound for conservative refinement.** "
        "The comparison uses the same 24 origin peptides as a balanced representative subset: eight starts from each source set "
        "(Sol, NF, and 1-Hem sets), with two starts sampled from each source-set composite-score quartile. Starts are ordered "
        "by origin S to avoid sorting by the observed exhaustive-search advantage. **a**, Composite score S = Sol + NF + 1-Hem for each origin, "
        "conservative agent output, and one-step fixed-length exhaustive search. Marker fill encodes method, marker edge "
        "encodes source set, and dashed lines show method means. **b**, Local-search headroom: the conservative-agent gain over origin, "
        "the fixed-length exhaustive-search gain over origin, and the remaining search-agent gap. **c**, Desired-direction property changes in "
        "Sol, NF, and 1-Hem with 95% bootstrap confidence intervals from paired starts. The fixed-length exhaustive search exceeds the agent "
        "output for 24/24 matched starts (one-sided sign-test P = "
        f"{stats_summary.iloc[0]['sign_test_p_greater']:.2e}; paired Wilcoxon P = {stats_summary.iloc[0]['wilcoxon_p_greater']:.2e}). "
        "This curve should be interpreted as a same-length local upper bound rather than a compute-matched baseline, because it "
        "exhaustively evaluates all single-substitution neighbors."
    )
    (OUT_DIR / "agent_vs_nonagent_oracle_framing_caption_v3.md").write_text(caption, encoding="utf-8")


def fig3_stats(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = records.copy()
    records["edit_distance"] = [
        hamming_or_levenshtein(a, b) for a, b in zip(records["original_sequence"], records["final_sequence"])
    ]
    breadth = (
        records.groupby(["method", "method_label"], sort=False)
        .agg(
            n=("start_id", "count"),
            length_changed_count=("length_changed", "sum"),
            delta_length_mean=("delta_length", "mean"),
            edit_distance_mean=("edit_distance", "mean"),
            edit_distance_sem=("edit_distance", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        )
        .reset_index()
    )
    aggressive = records[records["method"] == "aggressive"].copy()
    length_strata = []
    for label, subset in [
        ("Aggressive length-preserving", aggressive[aggressive["delta_length"] == 0]),
        ("Aggressive length-changing", aggressive[aggressive["delta_length"] != 0]),
        ("Aggressive all", aggressive),
        ("Greedy all", records[records["method"] == "greedy"]),
        ("Conservative all", records[records["method"] == "conservative"]),
    ]:
        values = subset["delta_score"].to_numpy(float)
        if len(values):
            low, high = bootstrap_ci(values, seed=420)
            length_strata.append(
                {
                    "group": label,
                    "n": len(values),
                    "mean_delta_S": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "improved_count": int((values > 0).sum()),
                }
            )
    head = pd.read_csv(FIG3_DIR / "aggressive_head_to_head.csv")
    diff_greedy = head["aggressive_minus_greedy"].to_numpy(float)
    head_stats = pd.DataFrame(
        [
            {
                "comparison": "aggressive_minus_greedy",
                "n": len(diff_greedy),
                "mean": float(diff_greedy.mean()),
                "median": float(np.median(diff_greedy)),
                "aggressive_higher_count": int((diff_greedy > 0).sum()),
                "sign_test_p_greater": float(stats.binomtest(int((diff_greedy > 0).sum()), len(diff_greedy), 0.5, alternative="greater").pvalue),
                "wilcoxon_p_greater": float(stats.wilcoxon(diff_greedy, alternative="greater", zero_method="wilcox").pvalue),
                "paired_effect_size_dz": paired_effect(diff_greedy),
            }
        ]
    )
    return breadth, pd.DataFrame(length_strata), head_stats


def plot_fig3() -> None:
    records = pd.read_csv(FIG3_DIR / "aggressive_comparison_matched_records.csv")
    summary = pd.read_csv(FIG3_DIR / "aggressive_summary_stats.csv")
    breadth, length_strata, head_stats = fig3_stats(records)
    records["edit_distance"] = [
        hamming_or_levenshtein(a, b) for a, b in zip(records["original_sequence"], records["final_sequence"])
    ]

    method_order = ["conservative", "greedy", "aggressive"]
    method_labels = {"conservative": "Conservative", "greedy": "Fixed-length exhaustive search", "aggressive": "Aggressive"}

    fig = plt.figure(figsize=(7.55, 5.7), constrained_layout=False)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.2, 0.95, 0.95], height_ratios=[1.0, 0.95], wspace=0.45, hspace=0.62)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_e = fig.add_subplot(grid[1, 2])

    starts = sorted(records["start_id"].unique())
    x_positions = {"origin": 0, "conservative": 1, "greedy": 2, "aggressive": 3.35}
    rng = np.random.default_rng(12)
    ax_a.axvspan(-0.35, 2.45, color="#F3F3F3", zorder=0)
    ax_a.axvspan(2.65, 3.75, color="#F9EFF2", zorder=0)
    ax_a.text(1.0, 2.066, "fixed-length regime", ha="center", va="bottom", fontsize=6.1, color="#555555")
    ax_a.text(3.35, 2.078, "expanded-action\nregime", ha="center", va="bottom", fontsize=6.1, color="#7A3E4B")
    for start in starts:
        sub = records[records["start_id"] == start].set_index("method")
        ys = [sub.loc["conservative", "original_score"], sub.loc["conservative", "final_score"], sub.loc["greedy", "final_score"], sub.loc["aggressive", "final_score"]]
        xs = [0, 1, 2, 3.35]
        ax_a.plot(xs, ys, color="#D7D7D7", lw=0.7, zorder=1)
    ax_a.scatter(np.zeros(len(starts)) + rng.normal(0, 0.025, len(starts)), records[records["method"] == "conservative"]["original_score"], s=16, color="#222222", alpha=0.85, zorder=3, label="Origin")
    for method in method_order:
        sub = records[records["method"] == method]
        ax_a.scatter(np.full(len(sub), x_positions[method]) + rng.normal(0, 0.025, len(sub)), sub["final_score"], s=18, color=METHOD_COLORS_F3[method], alpha=0.9, zorder=4, label=method_labels[method])
    means = [records[records["method"] == "conservative"]["original_score"].mean()] + [records[records["method"] == m]["final_score"].mean() for m in method_order]
    ax_a.plot([0, 1, 2, 3.35], means, color="black", lw=1.4, marker="D", markersize=4, zorder=6)
    ax_a.axvline(2.55, color="#B8B8B8", lw=0.8, ls="--")
    ax_a.set_xticks([0, 1, 2, 3.35])
    ax_a.set_xticklabels(["Origin", "Cons.\nagent", "Fixed-len.\nexhaustive\nsearch", "Agg.\nagent"])
    ax_a.set_xlim(-0.55, 3.85)
    ax_a.set_ylabel("Composite score S")
    ax_a.set_title("Matched-start outputs by search regime", loc="left", pad=6)
    ax_a.grid(axis="y", color="#E5E5E5", lw=0.6)
    ax_a.legend(loc="lower right", fontsize=6.2)
    add_panel_label(ax_a, "a")

    pool_order = [("sol", "Sol set"), ("nf", "NF set"), ("hem", "1-Hem set"), ("overall", "All")]
    xpos = np.arange(len(pool_order))
    width = 0.24
    for j, method in enumerate(method_order):
        means, lows, highs = [], [], []
        for pool, _ in pool_order:
            sub = records[records["method"] == method] if pool == "overall" else records[(records["method"] == method) & (records["source_pool"] == pool)]
            values = sub["delta_score"].to_numpy(float)
            low, high = bootstrap_ci(values, seed=500 + j)
            means.append(values.mean())
            lows.append(values.mean() - low)
            highs.append(high - values.mean())
        ax_b.bar(xpos + (j - 1) * width, means, width=width, color=METHOD_COLORS_F3[method], label=method_labels[method])
        ax_b.errorbar(xpos + (j - 1) * width, means, yerr=[lows, highs], fmt="none", ecolor="#202020", lw=0.8, capsize=2)
    ax_b.axhline(0, color="#404040", lw=0.8)
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels([p[1] for p in pool_order], rotation=25, ha="right")
    ax_b.set_ylabel("Mean Delta S")
    ax_b.set_title("Composite-score gains by regime", loc="left", pad=6)
    ax_b.grid(axis="y", color="#E5E5E5", lw=0.6)
    add_panel_label(ax_b, "b")

    prop_order = [("delta_solubility", "Sol"), ("delta_non_fouling", "NF"), ("delta_antihemolysis", "1-Hem")]
    xpos = np.arange(len(prop_order))
    for j, method in enumerate(method_order):
        means, lows, highs = [], [], []
        for col, _ in prop_order:
            values = records[records["method"] == method][col].to_numpy(float)
            low, high = bootstrap_ci(values, seed=610 + j)
            means.append(values.mean())
            lows.append(values.mean() - low)
            highs.append(high - values.mean())
        ax_c.bar(xpos + (j - 1) * width, means, width=width, color=METHOD_COLORS_F3[method])
        ax_c.errorbar(xpos + (j - 1) * width, means, yerr=[lows, highs], fmt="none", ecolor="#202020", lw=0.8, capsize=2)
    ax_c.axhline(0, color="#404040", lw=0.8)
    ax_c.set_xticks(xpos)
    ax_c.set_xticklabels([p[1] for p in prop_order])
    ax_c.set_ylabel("Mean property change")
    ax_c.set_title("Property-direction changes", loc="left", pad=6)
    ax_c.grid(axis="y", color="#E5E5E5", lw=0.6)
    add_panel_label(ax_c, "c")

    ax_d.axis("off")
    ax_d.set_title("Edit-policy breadth", loc="left", pad=6)
    add_panel_label(ax_d, "d")
    header = ["Method", "Len\nchanged", "Mean\nDelta len", "Mean\nedit"]
    xs = [0.02, 0.45, 0.70, 0.94]
    y0 = 0.83
    for x, h in zip(xs, header):
        ax_d.text(x, y0, h, transform=ax_d.transAxes, ha="center" if x > 0.1 else "left", va="center", fontsize=5.8, fontweight="bold")
    ax_d.plot([0.02, 0.98], [y0 - 0.06, y0 - 0.06], transform=ax_d.transAxes, color="#D7D7D7", lw=0.7)
    table_method_labels = {"conservative": "Conservative", "greedy": "Exhaustive", "aggressive": "Aggressive"}
    for i, method in enumerate(method_order):
        row = breadth[breadth["method"] == method].iloc[0]
        y = y0 - (i + 1) * 0.20
        vals = [table_method_labels[method], f"{int(row['length_changed_count'])}/{int(row['n'])}", f"{row['delta_length_mean']:+.2f}", f"{row['edit_distance_mean']:.2f}"]
        for x, val in zip(xs, vals):
            ax_d.text(x, y, val, transform=ax_d.transAxes, ha="center" if x > 0.1 else "left", va="center", fontsize=6.1)

    groups = ["Conservative all", "Greedy all", "Aggressive length-preserving", "Aggressive length-changing", "Aggressive all"]
    subset = length_strata.set_index("group").loc[groups].reset_index()
    x = np.arange(len(subset))
    colors = ["#7884B4", "#767676", "#D8A0AD", "#C76F82", "#C76F82"]
    ax_e.bar(x, subset["mean_delta_S"], color=colors, width=0.68)
    lows = subset["mean_delta_S"] - subset["ci95_low"]
    highs = subset["ci95_high"] - subset["mean_delta_S"]
    ax_e.errorbar(x, subset["mean_delta_S"], yerr=[lows, highs], fmt="none", ecolor="#202020", lw=0.8, capsize=2)
    for xi, (_, row) in enumerate(subset.iterrows()):
        ax_e.text(xi, row["ci95_high"] + 0.008, f"n={int(row['n'])}", ha="center", va="top", fontsize=6.1)
    ax_e.axhline(0, color="#404040", lw=0.8)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(["Cons.", "Fixed-len.\nexhaustive", "Agg.\nequal-len", "Agg.\nlen-change", "Agg.\nall"], rotation=30, ha="right")
    ax_e.set_ylabel("Mean Delta S")
    ax_e.set_title("Aggressive gain is length-coupled", loc="left", pad=6)
    ax_e.grid(axis="y", color="#E5E5E5", lw=0.6)
    add_panel_label(ax_e, "e")

    fig.suptitle("Aggressive refinement gains arise from expanded action space", y=0.995, fontsize=10.0)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.14)
    save_figure(fig, OUT_DIR / "aggressive_refinement_action_space_v3")

    breadth.to_csv(OUT_DIR / "aggressive_edit_policy_breadth_v3.csv", index=False)
    length_strata.to_csv(OUT_DIR / "aggressive_length_stratified_deltaS_v3.csv", index=False)
    head_stats.to_csv(OUT_DIR / "aggressive_head_to_head_stats_v3.csv", index=False)
    summary.to_csv(OUT_DIR / "aggressive_original_summary_stats_v3.csv", index=False)

    caption = (
        "**Figure Y | Aggressive refinement gains arise from expanded action space.** "
        "All panels use the same 24 matched starts as the conservative-agent and greedy-control comparison "
        "(8 per source set; 2 per source-set score quartile). Composite score is S = Sol + NF + 1-Hem. "
        "**a**, Matched-start final scores for origin, conservative agent, fixed-length exhaustive search, and aggressive agent; "
        "grey lines connect outputs derived from the same origin and do not indicate a sequential pipeline. The shaded regions separate "
        "the fixed-length regime from the expanded-action regime. **b**, Mean Delta S "
        "by source set and overall with 95% bootstrap confidence intervals. **c**, Mean desired-direction changes in Sol, NF, "
        "and 1-Hem. **d**, Edit-policy breadth, including starts with changed length, mean length change, and mean Levenshtein "
        "edit distance. **e**, Length-coupling diagnostic showing aggressive-agent Delta S stratified by whether sequence length "
        "changed. Aggressive refinement exceeds fixed-length exhaustive search for 22/24 starts, but 21/24 aggressive outputs change length "
        f"(mean length change {breadth[breadth['method']=='aggressive']['delta_length_mean'].iloc[0]:+.2f} aa), so its gains are not directly comparable "
        "to fixed-length baselines. The length-preserving aggressive subset is small and should be treated as a diagnostic rather than a definitive control; "
        "a length-matched random or permutation control remains necessary for a full specification-gaming test."
    )
    (OUT_DIR / "aggressive_refinement_action_space_caption_v3.md").write_text(caption, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    plot_fig2()
    plot_fig3()
    print(f"Wrote revised figures and statistics to {OUT_DIR}")


if __name__ == "__main__":
    main()
