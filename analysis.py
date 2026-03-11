#!/usr/bin/env python3
"""
Statistical aggregation and publication-quality plotting.

Reads results from the `results/` directory tree produced by run_all.py,
aggregates across seeds, and generates figures + a text report for the paper.

Usage
-----
    # Generate everything
    python analysis.py results/ --output figures/

    # Generate only specific plots
    python analysis.py results/ --output figures/ --plot main
    python analysis.py results/ --output figures/ --plot lambda
    python analysis.py results/ --output figures/ --plot depth
    python analysis.py results/ --output figures/ --plot noise
    python analysis.py results/ --output figures/ --plot size
    python analysis.py results/ --output figures/ --plot report
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import indent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLT_STYLE = "seaborn-v0_8-paper"

# Known mu_true values indexed by dataset name
MU_TRUE_MAP = {
    "D1_d2_w20":   4.0,
    "D2_d1.5_w30": 3.0,
    "D3_d3_w30":   6.0,
    "D4_d4_w40":   8.0,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_metrics(root: str | Path) -> pd.DataFrame:
    """Recursively load all metrics.csv files under *root*.

    Directory structure expected:
        <sweep_tag>/<model>/<dataset>/seed_<N>/metrics.csv

    For the main sweep the structure is:
        main/<model>/<dataset>/seed_<N>/metrics.csv

    Every row gets: model, dataset, seed, sweep_tag columns.
    Also reads config.json to attach mu_true when available.
    """
    root = Path(root)
    frames = []
    for csv_path in sorted(root.rglob("metrics.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        parts = csv_path.relative_to(root).parts
        seed_dir = parts[-2]
        seed_val = int(seed_dir.split("_")[-1])
        dataset_name = parts[-3]
        model_name = parts[-4]
        sweep_tag = "/".join(parts[:-4]) if len(parts) > 4 else "main"

        df["model"] = model_name
        df["dataset"] = dataset_name
        df["seed"] = seed_val
        df["sweep_tag"] = sweep_tag

        # Attach mu_true from known map or from config.json
        mu_true = MU_TRUE_MAP.get(dataset_name)
        config_path = csv_path.parent / "config.json"
        if mu_true is None and config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                d = cfg.get("dataset", {}).get("d")
                if d is not None:
                    mu_true = 2 * d
            except Exception:
                pass
        df["mu_true"] = mu_true
        df["mu_error"] = (df["mu"] - mu_true).abs() if mu_true is not None else np.nan

        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No metrics.csv found under {root}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_final_metrics(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Mean & std of final-step metrics across seeds."""
    if group_cols is None:
        group_cols = ["model", "dataset", "sweep_tag"]

    last = df.groupby(["model", "dataset", "seed", "sweep_tag"]).tail(1)
    agg = last.groupby(group_cols).agg(
        loss_mean=("loss", "mean"),       loss_std=("loss", "std"),
        loss_phys_mean=("loss_phys", "mean"), loss_phys_std=("loss_phys", "std"),
        loss_data_mean=("loss_data", "mean"), loss_data_std=("loss_data", "std"),
        mu_mean=("mu", "mean"),           mu_std=("mu", "std"),
        mu_error_mean=("mu_error", "mean"), mu_error_std=("mu_error", "std"),
        step_time_mean=("step_time_s", "mean"), step_time_std=("step_time_s", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return agg


def aggregate_curves(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Mean & std of metrics at every step across seeds."""
    if group_cols is None:
        group_cols = ["model", "dataset", "sweep_tag", "step"]

    return df.groupby(group_cols).agg(
        loss_mean=("loss", "mean"),       loss_std=("loss", "std"),
        loss_phys_mean=("loss_phys", "mean"), loss_phys_std=("loss_phys", "std"),
        mu_mean=("mu", "mean"),           mu_std=("mu", "std"),
        mu_error_mean=("mu_error", "mean"), mu_error_std=("mu_error", "std"),
    ).reset_index()


def compute_convergence_step(df: pd.DataFrame, threshold_factor: float = 0.05) -> pd.DataFrame:
    """For each run, find the first step where loss ≤ factor × initial_loss.

    Returns a DataFrame with columns [model, dataset, seed, sweep_tag,
    convergence_step, initial_loss, final_loss].
    """
    records = []
    for (model, dataset, seed, sweep_tag), g in df.groupby(
        ["model", "dataset", "seed", "sweep_tag"]
    ):
        g = g.sort_values("step")
        initial = g["loss"].iloc[0]
        threshold = threshold_factor * initial
        crossed = g[g["loss"] <= threshold]
        conv_step = int(crossed["step"].iloc[0]) if not crossed.empty else None
        records.append({
            "model": model, "dataset": dataset, "seed": seed, "sweep_tag": sweep_tag,
            "convergence_step": conv_step,
            "initial_loss": initial,
            "final_loss": g["loss"].iloc[-1],
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _apply_style():
    try:
        plt.style.use(PLT_STYLE)
    except OSError:
        try:
            plt.style.use("seaborn-v0_8")
        except OSError:
            pass


def _save_fig(fig: plt.Figure, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


MODEL_ORDER = ["PIML_113", "PIML_2209", "PIQML_109"]
MODEL_COLORS = {"PIML_113": "steelblue", "PIML_2209": "darkorange", "PIQML_109": "seagreen"}


# ---------------------------------------------------------------------------
# Main result plots (Concern #5 -- statistical measures)
# ---------------------------------------------------------------------------

def plot_loss_curves(df: pd.DataFrame, output_dir: Path, sweep_tag: str = "main"):
    """Loss convergence with shaded ±1 std -- addresses Concern #5."""
    _apply_style()
    sub = df[df["sweep_tag"] == sweep_tag]
    curves = aggregate_curves(sub)

    for ds_name, ds_df in curves.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for model_name in MODEL_ORDER:
            m_df = ds_df[ds_df["model"] == model_name].sort_values("step")
            if m_df.empty:
                continue
            color = MODEL_COLORS.get(model_name)
            ax.plot(m_df["step"], m_df["loss_mean"], label=model_name, color=color)
            if m_df["loss_std"].notna().any():
                ax.fill_between(
                    m_df["step"],
                    (m_df["loss_mean"] - m_df["loss_std"]).clip(lower=0),
                    m_df["loss_mean"] + m_df["loss_std"],
                    alpha=0.18, color=color,
                )
        ax.set_xlabel("Training step")
        ax.set_ylabel("Combined loss (log scale)")
        ax.set_yscale("log")
        ax.set_title(f"Loss convergence  [{ds_name}]  (mean ± 1 std, {sub['seed'].nunique()} seeds)")
        ax.legend()
        fig.tight_layout()
        _save_fig(fig, output_dir, f"loss_curve_{ds_name}")


def plot_mu_recovery(df: pd.DataFrame, output_dir: Path, sweep_tag: str = "main"):
    """Mu recovery curves with shaded std -- addresses Concern #5."""
    _apply_style()
    sub = df[df["sweep_tag"] == sweep_tag]
    curves = aggregate_curves(sub)

    for ds_name, ds_df in curves.groupby("dataset"):
        mu_true = MU_TRUE_MAP.get(ds_name)
        fig, ax = plt.subplots(figsize=(9, 5))
        if mu_true is not None:
            ax.axhline(mu_true, color="red", linestyle="--", linewidth=1.5,
                       label=f"True $\\mu$ = {mu_true}")
        for model_name in MODEL_ORDER:
            m_df = ds_df[ds_df["model"] == model_name].sort_values("step")
            if m_df.empty:
                continue
            color = MODEL_COLORS.get(model_name)
            ax.plot(m_df["step"], m_df["mu_mean"], label=model_name, color=color)
            if m_df["mu_std"].notna().any():
                ax.fill_between(
                    m_df["step"],
                    m_df["mu_mean"] - m_df["mu_std"],
                    m_df["mu_mean"] + m_df["mu_std"],
                    alpha=0.18, color=color,
                )
        ax.set_xlabel("Training step")
        ax.set_ylabel("Estimated $\\mu$")
        ax.set_title(f"Parameter recovery  [{ds_name}]  (mean ± 1 std)")
        ax.legend()
        fig.tight_layout()
        _save_fig(fig, output_dir, f"mu_recovery_{ds_name}")


def plot_final_comparison_bars(df: pd.DataFrame, output_dir: Path, sweep_tag: str = "main"):
    """Bar chart: final loss and mu error per model -- addresses Concern #5."""
    _apply_style()
    agg = aggregate_final_metrics(df[df["sweep_tag"] == sweep_tag])

    for ds_name, ds_df in agg.groupby("dataset"):
        ordered = [m for m in MODEL_ORDER if m in ds_df["model"].values]
        ds_df = ds_df.set_index("model").reindex(ordered).reset_index()
        x = np.arange(len(ordered))
        colors = [MODEL_COLORS.get(m, "grey") for m in ordered]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].bar(x, ds_df["loss_mean"], yerr=ds_df["loss_std"].fillna(0),
                    color=colors, capsize=5, alpha=0.8)
        axes[0].set_xticks(x); axes[0].set_xticklabels(ordered, rotation=15)
        axes[0].set_ylabel("Final loss"); axes[0].set_title(f"Final combined loss [{ds_name}]")

        axes[1].bar(x, ds_df["mu_error_mean"], yerr=ds_df["mu_error_std"].fillna(0),
                    color=colors, capsize=5, alpha=0.8)
        axes[1].set_xticks(x); axes[1].set_xticklabels(ordered, rotation=15)
        axes[1].set_ylabel("|μ_pred − μ_true|"); axes[1].set_title(f"μ recovery error [{ds_name}]")

        axes[2].bar(x, ds_df["step_time_mean"] * 1000,
                    yerr=ds_df["step_time_std"].fillna(0) * 1000,
                    color=colors, capsize=5, alpha=0.8)
        axes[2].set_xticks(x); axes[2].set_xticklabels(ordered, rotation=15)
        axes[2].set_ylabel("ms / step"); axes[2].set_title(f"Cost per step [{ds_name}]")

        fig.suptitle(f"Summary  [{ds_name}]  (mean ± 1 std, n={ds_df['n_seeds'].max()} seeds)")
        fig.tight_layout()
        _save_fig(fig, output_dir, f"bar_comparison_{ds_name}")


# ---------------------------------------------------------------------------
# Lambda sensitivity (Concern #3)
# ---------------------------------------------------------------------------

def plot_lambda_sensitivity(df: pd.DataFrame, output_dir: Path):
    """Heatmap of final loss vs λ3 × λ4 per model/dataset."""
    _apply_style()
    lambda_df = df[df["sweep_tag"].str.contains("l3_", na=False)]
    if lambda_df.empty:
        print("  [lambda] No lambda sweep data found -- skipping.")
        return

    def _parse_tag(tag):
        try:
            l3 = float(tag.split("l3_")[1].split("_l4")[0])
            l4 = float(tag.split("l4_")[1].split("/")[0])
            return l3, l4
        except Exception:
            return None, None

    lambda_df = lambda_df.copy()
    lambda_df[["l3", "l4"]] = lambda_df["sweep_tag"].apply(
        lambda t: pd.Series(_parse_tag(t))
    )
    lambda_df = lambda_df.dropna(subset=["l3", "l4"])

    agg = lambda_df.groupby(["model", "dataset", "l3", "l4"]).agg(
        loss_mean=("loss", "mean")
    ).reset_index()

    for (model_name, ds_name), g in agg.groupby(["model", "dataset"]):
        l3_vals = sorted(g["l3"].unique())
        l4_vals = sorted(g["l4"].unique())
        grid = np.full((len(l3_vals), len(l4_vals)), np.nan)
        for _, row in g.iterrows():
            i = l3_vals.index(row["l3"])
            j = l4_vals.index(row["l4"])
            grid[i, j] = row["loss_mean"]

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(np.log10(grid + 1e-10), aspect="auto", origin="lower", cmap="viridis_r")
        ax.set_xticks(range(len(l4_vals)))
        ax.set_xticklabels([f"{v:.0e}" for v in l4_vals], rotation=45)
        ax.set_yticks(range(len(l3_vals)))
        ax.set_yticklabels([f"{v:.1e}" for v in l3_vals])
        ax.set_xlabel("$\\lambda_4$ (data weight)")
        ax.set_ylabel("$\\lambda_3$ (physics weight)")
        ax.set_title(f"Final loss (log10)  [{model_name} / {ds_name}]")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("log10(loss)")
        fig.tight_layout()
        _save_fig(fig, output_dir, f"lambda_heatmap_{model_name}_{ds_name}")


# ---------------------------------------------------------------------------
# Circuit depth ablation (Concern #3)
# ---------------------------------------------------------------------------

def plot_depth_ablation(df: pd.DataFrame, output_dir: Path):
    """Loss and mu error vs circuit depth."""
    _apply_style()
    depth_df = df[df["model"].str.startswith("PIQML_L")].copy()
    if depth_df.empty:
        print("  [depth] No depth sweep data found -- skipping.")
        return

    depth_df["depth"] = depth_df["model"].str.replace("PIQML_L", "").astype(int)
    agg = depth_df.groupby(["depth", "dataset", "seed"]).tail(1).groupby(
        ["depth", "dataset"]
    ).agg(
        loss_mean=("loss", "mean"), loss_std=("loss", "std"),
        mu_error_mean=("mu_error", "mean"), mu_error_std=("mu_error", "std"),
        step_time_mean=("step_time_s", "mean"),
    ).reset_index()

    for ds_name, ds_df in agg.groupby("dataset"):
        ds_df = ds_df.sort_values("depth")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].errorbar(ds_df["depth"], ds_df["loss_mean"], yerr=ds_df["loss_std"].fillna(0),
                         marker="o", capsize=5, color="seagreen")
        axes[0].set_xlabel("Circuit depth (layers)")
        axes[0].set_ylabel("Final loss"); axes[0].set_title(f"Loss vs. depth [{ds_name}]")

        axes[1].errorbar(ds_df["depth"], ds_df["mu_error_mean"],
                         yerr=ds_df["mu_error_std"].fillna(0),
                         marker="o", capsize=5, color="seagreen")
        axes[1].set_xlabel("Circuit depth (layers)")
        axes[1].set_ylabel("|μ_pred − μ_true|"); axes[1].set_title(f"μ error vs. depth [{ds_name}]")

        axes[2].plot(ds_df["depth"], ds_df["step_time_mean"] * 1000,
                     marker="o", color="seagreen")
        axes[2].set_xlabel("Circuit depth (layers)")
        axes[2].set_ylabel("ms / step"); axes[2].set_title(f"Cost vs. depth [{ds_name}]")

        fig.tight_layout()
        _save_fig(fig, output_dir, f"depth_ablation_{ds_name}")


# ---------------------------------------------------------------------------
# Component ablation (Concern #3)
# ---------------------------------------------------------------------------

def plot_component_ablation(df: pd.DataFrame, output_dir: Path):
    """Final loss and mu error by loss component variant."""
    _apply_style()
    comp_df = df[df["sweep_tag"].str.startswith("component/", na=False)].copy()
    if comp_df.empty:
        print("  [component] No component sweep data found -- skipping.")
        return

    comp_df["variant"] = comp_df["sweep_tag"].str.replace("component/", "")
    agg = aggregate_final_metrics(comp_df, group_cols=["model", "dataset", "variant"])

    for ds_name, ds_df in agg.groupby("dataset"):
        variants = ds_df["variant"].unique()
        ordered = [m for m in MODEL_ORDER if m in ds_df["model"].values]
        x = np.arange(len(ordered))
        width = 0.25
        colors_var = {"physics_only": "steelblue", "data_only": "darkorange", "combined": "seagreen"}

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for vi, variant in enumerate(["physics_only", "data_only", "combined"]):
            sub = ds_df[ds_df["variant"] == variant].set_index("model").reindex(ordered)
            offset = (vi - 1) * width
            axes[0].bar(x + offset, sub["loss_mean"].fillna(0),
                        width=width, label=variant, color=colors_var.get(variant),
                        yerr=sub["loss_std"].fillna(0), capsize=3, alpha=0.8)
            axes[1].bar(x + offset, sub["mu_error_mean"].fillna(0),
                        width=width, label=variant, color=colors_var.get(variant),
                        yerr=sub["mu_error_std"].fillna(0), capsize=3, alpha=0.8)

        for ax, ylabel, title in [
            (axes[0], "Final loss", f"Loss by component [{ds_name}]"),
            (axes[1], "|μ_pred − μ_true|", f"μ error by component [{ds_name}]"),
        ]:
            ax.set_xticks(x); ax.set_xticklabels(ordered, rotation=15)
            ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()

        fig.tight_layout()
        _save_fig(fig, output_dir, f"component_ablation_{ds_name}")


# ---------------------------------------------------------------------------
# Noise sensitivity (Concern #4)
# ---------------------------------------------------------------------------

def plot_noise_sensitivity(df: pd.DataFrame, output_dir: Path):
    """Final loss and mu error vs noise level."""
    _apply_style()
    noise_df = df[df["sweep_tag"].str.startswith("noise/", na=False)].copy()
    if noise_df.empty:
        print("  [noise] No noise sweep data found -- skipping.")
        return

    noise_df["noise_std"] = noise_df["sweep_tag"].str.extract(r"noise_([\d.]+)").astype(float)
    agg = noise_df.groupby(["model", "dataset", "noise_std", "seed"]).tail(1).groupby(
        ["model", "dataset", "noise_std"]
    ).agg(
        loss_mean=("loss", "mean"), loss_std=("loss", "std"),
        mu_error_mean=("mu_error", "mean"), mu_error_std=("mu_error", "std"),
    ).reset_index()

    for ds_name, ds_df in agg.groupby("dataset"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for model_name in MODEL_ORDER:
            m_df = ds_df[ds_df["model"] == model_name].sort_values("noise_std")
            if m_df.empty:
                continue
            color = MODEL_COLORS.get(model_name)
            axes[0].errorbar(m_df["noise_std"], m_df["loss_mean"],
                             yerr=m_df["loss_std"].fillna(0),
                             marker="o", capsize=4, label=model_name, color=color)
            axes[1].errorbar(m_df["noise_std"], m_df["mu_error_mean"],
                             yerr=m_df["mu_error_std"].fillna(0),
                             marker="o", capsize=4, label=model_name, color=color)

        for ax, ylabel, title in [
            (axes[0], "Final loss", f"Loss vs. noise  [{ds_name}]"),
            (axes[1], "|μ_pred − μ_true|", f"μ error vs. noise  [{ds_name}]"),
        ]:
            ax.set_xlabel("Noise std")
            ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()

        fig.tight_layout()
        _save_fig(fig, output_dir, f"noise_sensitivity_{ds_name}")


# ---------------------------------------------------------------------------
# Training-set size sensitivity (Concern #4)
# ---------------------------------------------------------------------------

def plot_size_sensitivity(df: pd.DataFrame, output_dir: Path):
    """Final loss and mu error vs training-set fraction."""
    _apply_style()
    size_df = df[df["sweep_tag"].str.startswith("size/", na=False)].copy()
    if size_df.empty:
        print("  [size] No size sweep data found -- skipping.")
        return

    size_df["frac"] = size_df["sweep_tag"].str.extract(r"frac_([\d.]+)").astype(float)
    agg = size_df.groupby(["model", "dataset", "frac", "seed"]).tail(1).groupby(
        ["model", "dataset", "frac"]
    ).agg(
        loss_mean=("loss", "mean"), loss_std=("loss", "std"),
        mu_error_mean=("mu_error", "mean"), mu_error_std=("mu_error", "std"),
    ).reset_index()

    for ds_name, ds_df in agg.groupby("dataset"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for model_name in MODEL_ORDER:
            m_df = ds_df[ds_df["model"] == model_name].sort_values("frac")
            if m_df.empty:
                continue
            color = MODEL_COLORS.get(model_name)
            axes[0].errorbar(m_df["frac"], m_df["loss_mean"],
                             yerr=m_df["loss_std"].fillna(0),
                             marker="o", capsize=4, label=model_name, color=color)
            axes[1].errorbar(m_df["frac"], m_df["mu_error_mean"],
                             yerr=m_df["mu_error_std"].fillna(0),
                             marker="o", capsize=4, label=model_name, color=color)

        for ax, ylabel, title in [
            (axes[0], "Final loss", f"Loss vs. training fraction  [{ds_name}]"),
            (axes[1], "|μ_pred − μ_true|", f"μ error vs. training fraction  [{ds_name}]"),
        ]:
            ax.set_xlabel("Training fraction")
            ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()

        fig.tight_layout()
        _save_fig(fig, output_dir, f"size_sensitivity_{ds_name}")


# ---------------------------------------------------------------------------
# Statistical report (Concern #5)
# ---------------------------------------------------------------------------

def generate_statistical_report(df: pd.DataFrame, output_dir: Path) -> str:
    """Write a plain-text statistical report ready to copy into the paper."""
    lines = []

    def h(text):
        lines.append("\n" + "=" * 70)
        lines.append(f"  {text}")
        lines.append("=" * 70)

    def row(label, value):
        lines.append(f"  {label:<45s}  {value}")

    # --- Main results ---
    main_df = df[df["sweep_tag"] == "main"]
    if not main_df.empty:
        h("MAIN RESULTS  (mean ± std across seeds)")
        agg = aggregate_final_metrics(main_df)
        for _, r in agg.iterrows():
            n = int(r["n_seeds"])
            row(f"{r['model']} | {r['dataset']}",
                f"loss={r['loss_mean']:.2f}±{r['loss_std']:.2f}  "
                f"μ_err={r['mu_error_mean']:.3f}±{r['mu_error_std']:.3f}  "
                f"t/step={r['step_time_mean']*1000:.1f}ms  n={n}")

    # --- Convergence steps ---
    if not main_df.empty:
        h("CONVERGENCE  (first step where loss ≤ 5% of initial)")
        conv = compute_convergence_step(main_df, threshold_factor=0.05)
        conv_agg = conv.groupby(["model", "dataset"]).agg(
            conv_mean=("convergence_step", "mean"),
            conv_std=("convergence_step", "std"),
        ).reset_index()
        for _, r in conv_agg.iterrows():
            s = f"{r['conv_mean']:.0f}±{r['conv_std']:.0f}" if not np.isnan(r['conv_mean']) else "not reached"
            row(f"{r['model']} | {r['dataset']}", f"convergence step: {s}")

    # --- Timing ---
    if not main_df.empty:
        h("TIMING SUMMARY  (total training time per run)")
        timed = main_df.groupby(["model", "dataset", "seed"]).agg(
            total_time=("step_time_s", "sum")
        ).reset_index()
        timed_agg = timed.groupby(["model", "dataset"]).agg(
            t_mean=("total_time", "mean"), t_std=("total_time", "std")
        ).reset_index()
        for _, r in timed_agg.iterrows():
            row(f"{r['model']} | {r['dataset']}",
                f"{r['t_mean']/60:.1f} ± {r['t_std']/60:.1f} min")

    # --- Lambda sweep summary ---
    lambda_df = df[df["sweep_tag"].str.contains("l3_", na=False)]
    if not lambda_df.empty:
        h("LAMBDA SENSITIVITY  (best λ combination per model/dataset)")
        lambda_df = lambda_df.copy()

        def _parse(tag):
            try:
                l3 = float(tag.split("l3_")[1].split("_l4")[0])
                l4 = float(tag.split("l4_")[1].split("/")[0])
                return l3, l4
            except Exception:
                return None, None

        lambda_df[["l3", "l4"]] = lambda_df["sweep_tag"].apply(
            lambda t: pd.Series(_parse(t))
        )
        lambda_df = lambda_df.dropna(subset=["l3", "l4"])
        last = lambda_df.groupby(["model", "dataset", "seed", "l3", "l4"]).tail(1)
        best = last.groupby(["model", "dataset", "l3", "l4"]).agg(
            loss_mean=("loss", "mean")
        ).reset_index()
        idx = best.groupby(["model", "dataset"])["loss_mean"].idxmin()
        for _, r in best.loc[idx].iterrows():
            row(f"{r['model']} | {r['dataset']}",
                f"best: λ3={r['l3']:.1e} λ4={r['l4']:.1e}  loss={r['loss_mean']:.2f}")

    # --- Component ablation summary ---
    comp_df = df[df["sweep_tag"].str.startswith("component/", na=False)]
    if not comp_df.empty:
        h("COMPONENT ABLATION  (final loss by variant)")
        comp_df = comp_df.copy()
        comp_df["variant"] = comp_df["sweep_tag"].str.replace("component/", "")
        agg = aggregate_final_metrics(comp_df, group_cols=["model", "dataset", "variant"])
        for _, r in agg.iterrows():
            row(f"{r['model']} | {r['dataset']} | {r['variant']}",
                f"loss={r['loss_mean']:.2f}±{r['loss_std']:.2f}  "
                f"μ_err={r['mu_error_mean']:.3f}±{r['mu_error_std']:.3f}")

    report = "\n".join(lines)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "statistical_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to {report_path}")
    return report


def generate_summary_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Save a CSV summary of final metrics (main sweep only)."""
    main_df = df[df["sweep_tag"] == "main"] if "main" in df["sweep_tag"].values else df
    agg = aggregate_final_metrics(main_df)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_dir / "summary_table.csv", index=False)
    print(f"\nSummary table ({len(agg)} rows) -> {output_dir / 'summary_table.csv'}")
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PLOT_CHOICES = ["all", "main", "lambda", "depth", "component", "noise", "size", "report"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyse PIQML experiment results")
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--output", type=str, default="figures")
    parser.add_argument("--plot", type=str, default="all", choices=PLOT_CHOICES)
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir} ...")
    df = load_all_metrics(args.results_dir)
    print(f"  Loaded {len(df):,} rows  |  {df['seed'].nunique()} seeds  |  "
          f"models: {sorted(df['model'].unique())}  |  "
          f"datasets: {sorted(df['dataset'].unique())}")

    out = Path(args.output)

    if args.plot in ("all", "main"):
        print("\n[main] Generating main result plots ...")
        plot_loss_curves(df, out)
        plot_mu_recovery(df, out)
        plot_final_comparison_bars(df, out)
        generate_summary_table(df, out)

    if args.plot in ("all", "lambda"):
        print("\n[lambda] Generating lambda sensitivity plots ...")
        plot_lambda_sensitivity(df, out / "lambda")

    if args.plot in ("all", "depth"):
        print("\n[depth] Generating depth ablation plots ...")
        plot_depth_ablation(df, out / "depth")

    if args.plot in ("all", "component"):
        print("\n[component] Generating component ablation plots ...")
        plot_component_ablation(df, out / "component")

    if args.plot in ("all", "noise"):
        print("\n[noise] Generating noise sensitivity plots ...")
        plot_noise_sensitivity(df, out / "noise")

    if args.plot in ("all", "size"):
        print("\n[size] Generating size sensitivity plots ...")
        plot_size_sensitivity(df, out / "size")

    if args.plot in ("all", "report"):
        print("\n[report] Generating statistical report ...")
        generate_statistical_report(df, out)

    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()
