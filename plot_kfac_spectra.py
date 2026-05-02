#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _sym(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-2, -1))


def _quantiles(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    q = np.quantile(values, [0.1, 0.5, 0.9])
    return float(q[0]), float(q[1]), float(q[2])


def _safe_log10(values: np.ndarray, min_value: float = 1e-12) -> np.ndarray:
    return np.log10(np.maximum(values, min_value))


def _infer_layer_kind(a_dim: int, g_dim: int) -> str:
    ratio = g_dim / max(a_dim, 1)
    if ratio >= 3.2:
        return "mlp_fc1_like"
    if ratio <= 0.33:
        return "mlp_fc2_like"
    if 2.3 <= ratio <= 3.2:
        return "qkv_like"
    if 0.8 <= ratio <= 1.2:
        return "proj_like"
    return "other"


def _extract_modules_state(ckpt: dict) -> list[dict]:
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint is not a dict.")

    candidates = []
    if isinstance(ckpt.get("optimizer"), dict):
        candidates.append(ckpt["optimizer"].get("kfac_reduce_state"))
    candidates.append(ckpt.get("kfac_reduce_state"))

    state = None
    for c in candidates:
        if isinstance(c, dict) and isinstance(c.get("modules_state"), list):
            state = c
            break
    if state is None:
        raise KeyError("Could not find kfac_reduce_state.modules_state in checkpoint.")

    mods = state["modules_state"]
    if not mods:
        raise ValueError("kfac_reduce_state.modules_state is empty.")
    return mods


def _load_layers(
    ckpt_path: Path,
    topk: int,
    damping: float | None,
    rel_thresh: float,
) -> list[dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    modules_state = _extract_modules_state(ckpt)

    layers: list[dict] = []
    for layer_idx, m in enumerate(modules_state):
        if "A" not in m or "G" not in m:
            continue

        A = _sym(m["A"].float().cpu())
        G = _sym(m["G"].float().cpu())

        eig_a = torch.linalg.eigvalsh(A).cpu().numpy()
        eig_g = torch.linalg.eigvalsh(G).cpu().numpy()

        pos_a = eig_a[eig_a > 1e-12]
        pos_g = eig_g[eig_g > 1e-12]
        top_a = np.sort(pos_a)[-min(topk, pos_a.size):] if pos_a.size else np.array([], dtype=np.float64)
        top_g = np.sort(pos_g)[-min(topk, pos_g.size):] if pos_g.size else np.array([], dtype=np.float64)

        mean_a = float(torch.trace(A).item() / A.shape[0])
        mean_g = float(torch.trace(G).item() / G.shape[0])

        a_damp_bal = math.nan
        g_damp_bal = math.nan
        a_top_bal = np.array([], dtype=np.float64)
        g_top_bal = np.array([], dtype=np.float64)
        a_top_scalar = np.array([], dtype=np.float64)
        g_top_scalar = np.array([], dtype=np.float64)

        if damping is not None and damping > 0.0:
            pi = math.sqrt(max(mean_a, 1e-12) / max(mean_g, 1e-12))
            root = math.sqrt(damping)
            a_damp_bal = pi * root
            g_damp_bal = root / pi
            a_top_bal = top_a / max(a_damp_bal, 1e-12)
            g_top_bal = top_g / max(g_damp_bal, 1e-12)
            a_top_scalar = top_a / damping
            g_top_scalar = top_g / damping

        thr_a = max(rel_thresh * (float(np.max(pos_a)) if pos_a.size else 0.0), 1e-12)
        thr_g = max(rel_thresh * (float(np.max(pos_g)) if pos_g.size else 0.0), 1e-12)
        nz_a = pos_a[pos_a > thr_a] if pos_a.size else np.array([], dtype=np.float64)
        nz_g = pos_g[pos_g > thr_g] if pos_g.size else np.array([], dtype=np.float64)

        layer = {
            "layer_idx": layer_idx,
            "a_dim": int(A.shape[0]),
            "g_dim": int(G.shape[0]),
            "ratio_g_over_a": float(G.shape[0] / max(A.shape[0], 1)),
            "kind": _infer_layer_kind(int(A.shape[0]), int(G.shape[0])),
            "mean_a": mean_a,
            "mean_g": mean_g,
            "a_damp_bal": a_damp_bal,
            "g_damp_bal": g_damp_bal,
            "eig_a_pos": pos_a,
            "eig_g_pos": pos_g,
            "top_a": top_a,
            "top_g": top_g,
            "top_a_bal": a_top_bal,
            "top_g_bal": g_top_bal,
            "top_a_scalar": a_top_scalar,
            "top_g_scalar": g_top_scalar,
            "nz_a": nz_a,
            "nz_g": nz_g,
        }
        layers.append(layer)

    if not layers:
        raise ValueError("No valid KFAC layers with A/G were found in checkpoint.")
    return layers


def _write_layer_summary(layers: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer_idx",
                "kind",
                "a_dim",
                "g_dim",
                "g_over_a",
                "mean_a",
                "mean_g",
                "a_damp_bal",
                "g_damp_bal",
                "a_top_q10",
                "a_top_q50",
                "a_top_q90",
                "g_top_q10",
                "g_top_q50",
                "g_top_q90",
                "a_top_bal_q50",
                "g_top_bal_q50",
                "a_top_scalar_q50",
                "g_top_scalar_q50",
                "nz_a_count",
                "nz_g_count",
            ]
        )
        for row in layers:
            a_q = _quantiles(row["top_a"])
            g_q = _quantiles(row["top_g"])
            a_bal_q = _quantiles(row["top_a_bal"])
            g_bal_q = _quantiles(row["top_g_bal"])
            a_sc_q = _quantiles(row["top_a_scalar"])
            g_sc_q = _quantiles(row["top_g_scalar"])
            writer.writerow(
                [
                    row["layer_idx"],
                    row["kind"],
                    row["a_dim"],
                    row["g_dim"],
                    row["ratio_g_over_a"],
                    row["mean_a"],
                    row["mean_g"],
                    row["a_damp_bal"],
                    row["g_damp_bal"],
                    a_q[0],
                    a_q[1],
                    a_q[2],
                    g_q[0],
                    g_q[1],
                    g_q[2],
                    a_bal_q[1],
                    g_bal_q[1],
                    a_sc_q[1],
                    g_sc_q[1],
                    row["nz_a"].size,
                    row["nz_g"].size,
                ]
            )


def _write_group_summary(layers: list[dict], out_csv: Path) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in layers:
        key = f'{row["kind"]}|A{row["a_dim"]}_G{row["g_dim"]}'
        grouped[key].append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group",
                "num_layers",
                "a_top_bal_q10",
                "a_top_bal_q50",
                "a_top_bal_q90",
                "g_top_bal_q10",
                "g_top_bal_q50",
                "g_top_bal_q90",
                "a_top_scalar_q50",
                "g_top_scalar_q50",
            ]
        )
        for key, rows in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            a_bal = np.concatenate([r["top_a_bal"] for r in rows if r["top_a_bal"].size > 0]) if rows else np.array([])
            g_bal = np.concatenate([r["top_g_bal"] for r in rows if r["top_g_bal"].size > 0]) if rows else np.array([])
            a_sc = np.concatenate([r["top_a_scalar"] for r in rows if r["top_a_scalar"].size > 0]) if rows else np.array([])
            g_sc = np.concatenate([r["top_g_scalar"] for r in rows if r["top_g_scalar"].size > 0]) if rows else np.array([])
            a_bal_q = _quantiles(a_bal)
            g_bal_q = _quantiles(g_bal)
            a_sc_q = _quantiles(a_sc)
            g_sc_q = _quantiles(g_sc)
            writer.writerow(
                [
                    key,
                    len(rows),
                    a_bal_q[0],
                    a_bal_q[1],
                    a_bal_q[2],
                    g_bal_q[0],
                    g_bal_q[1],
                    g_bal_q[2],
                    a_sc_q[1],
                    g_sc_q[1],
                ]
            )


def _plot_global_eig_hist(layers: list[dict], out_png: Path) -> None:
    all_a = np.concatenate([row["eig_a_pos"] for row in layers if row["eig_a_pos"].size > 0])
    all_g = np.concatenate([row["eig_g_pos"] for row in layers if row["eig_g_pos"].size > 0])
    top_a = np.concatenate([row["top_a"] for row in layers if row["top_a"].size > 0])
    top_g = np.concatenate([row["top_g"] for row in layers if row["top_g"].size > 0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(_safe_log10(all_a), bins=80, alpha=0.65, label="A (all eig > 0)")
    axes[0].hist(_safe_log10(all_g), bins=80, alpha=0.65, label="G (all eig > 0)")
    axes[0].set_title("Global Eigenvalue Distribution (log10)")
    axes[0].set_xlabel("log10(eigenvalue)")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].hist(_safe_log10(top_a), bins=60, alpha=0.65, label="A (top-k / layer)")
    axes[1].hist(_safe_log10(top_g), bins=60, alpha=0.65, label="G (top-k / layer)")
    axes[1].set_title("Top-k Eigenvalues per Layer (log10)")
    axes[1].set_xlabel("log10(eigenvalue)")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_damping_ratio_hist(layers: list[dict], out_png: Path) -> None:
    a_bal = np.concatenate([row["top_a_bal"] for row in layers if row["top_a_bal"].size > 0])
    g_bal = np.concatenate([row["top_g_bal"] for row in layers if row["top_g_bal"].size > 0])
    a_sc = np.concatenate([row["top_a_scalar"] for row in layers if row["top_a_scalar"].size > 0])
    g_sc = np.concatenate([row["top_g_scalar"] for row in layers if row["top_g_scalar"].size > 0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(_safe_log10(a_bal), bins=60, alpha=0.65, label="A / a_damp_bal")
    axes[0].hist(_safe_log10(g_bal), bins=60, alpha=0.65, label="G / g_damp_bal")
    axes[0].axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label="=1x damping")
    axes[0].set_title("Balanced Damping Ratios (top-k, log10)")
    axes[0].set_xlabel("log10(eig / balanced_damp)")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].hist(_safe_log10(a_sc), bins=60, alpha=0.65, label="A / scalar_damping")
    axes[1].hist(_safe_log10(g_sc), bins=60, alpha=0.65, label="G / scalar_damping")
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7, label="=1x damping")
    axes[1].set_title("Scalar Damping Ratios (top-k, log10)")
    axes[1].set_xlabel("log10(eig / scalar_damping)")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_layerwise_medians(layers: list[dict], out_png: Path) -> None:
    idx = np.array([row["layer_idx"] for row in layers], dtype=np.int64)
    a_med = np.array([_quantiles(row["top_a_bal"])[1] for row in layers], dtype=np.float64)
    g_med = np.array([_quantiles(row["top_g_bal"])[1] for row in layers], dtype=np.float64)
    a_q10 = np.array([_quantiles(row["top_a_bal"])[0] for row in layers], dtype=np.float64)
    a_q90 = np.array([_quantiles(row["top_a_bal"])[2] for row in layers], dtype=np.float64)
    g_q10 = np.array([_quantiles(row["top_g_bal"])[0] for row in layers], dtype=np.float64)
    g_q90 = np.array([_quantiles(row["top_g_bal"])[2] for row in layers], dtype=np.float64)

    order = np.argsort(idx)
    idx = idx[order]
    a_med, g_med = a_med[order], g_med[order]
    a_q10, a_q90 = a_q10[order], a_q90[order]
    g_q10, g_q90 = g_q10[order], g_q90[order]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(idx, a_med, label="A median(top-k / a_damp_bal)", color="tab:blue")
    axes[0].fill_between(idx, a_q10, a_q90, color="tab:blue", alpha=0.2, label="A q10-q90")
    axes[0].axhline(1.0, linestyle="--", color="k", linewidth=1.0, alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("ratio (log scale)")
    axes[0].set_title("Layerwise A vs Balanced Damping")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(idx, g_med, label="G median(top-k / g_damp_bal)", color="tab:orange")
    axes[1].fill_between(idx, g_q10, g_q90, color="tab:orange", alpha=0.2, label="G q10-q90")
    axes[1].axhline(1.0, linestyle="--", color="k", linewidth=1.0, alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("ratio (log scale)")
    axes[1].set_xlabel("layer index")
    axes[1].set_title("Layerwise G vs Balanced Damping")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_grouped_bars(layers: list[dict], out_png: Path, max_groups: int) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in layers:
        key = f'{row["kind"]}|A{row["a_dim"]}_G{row["g_dim"]}'
        grouped[key].append(row)

    items = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:max_groups]
    labels = []
    a_vals = []
    g_vals = []
    counts = []
    for key, rows in items:
        a_bal = np.concatenate([r["top_a_bal"] for r in rows if r["top_a_bal"].size > 0]) if rows else np.array([])
        g_bal = np.concatenate([r["top_g_bal"] for r in rows if r["top_g_bal"].size > 0]) if rows else np.array([])
        labels.append(key)
        a_vals.append(_quantiles(a_bal)[1])
        g_vals.append(_quantiles(g_bal)[1])
        counts.append(len(rows))

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(labels)), 5.3))
    ax.bar(x - w / 2, a_vals, width=w, label="A median(top-k / a_damp_bal)")
    ax.bar(x + w / 2, g_vals, width=w, label="G median(top-k / g_damp_bal)")
    for i, c in enumerate(counts):
        ax.text(i, max(a_vals[i], g_vals[i]) * 1.03 if np.isfinite(max(a_vals[i], g_vals[i])) else 1.0, f"n={c}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, linestyle="--", color="k", linewidth=1.0, alpha=0.7)
    ax.set_yscale("log")
    ax.set_ylabel("ratio (log scale)")
    ax.set_title("Balanced-Damping Strength by Layer Group")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot A/G eigenspectrum diagnostics from a KFAC-Muon checkpoint."
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint-*.pth.tar")
    parser.add_argument("--outdir", type=Path, default=Path("kfac_spectra"), help="Output directory for plots/csv")
    parser.add_argument("--topk", type=int, default=50, help="Top-k eigvals per layer for ratio analyses")
    parser.add_argument(
        "--damping",
        type=float,
        default=None,
        help="Scalar KFAC damping at this checkpoint epoch. Required for damping-ratio plots.",
    )
    parser.add_argument(
        "--rel-thresh",
        type=float,
        default=1e-6,
        help="Relative threshold (to layer max eig) to define nontrivial eigenvalues.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=10,
        help="Max number of layer groups to show in grouped bar plot.",
    )
    args = parser.parse_args()

    layers = _load_layers(
        ckpt_path=args.ckpt,
        topk=args.topk,
        damping=args.damping,
        rel_thresh=args.rel_thresh,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    _write_layer_summary(layers, args.outdir / "layer_summary.csv")
    _write_group_summary(layers, args.outdir / "group_summary.csv")
    _plot_global_eig_hist(layers, args.outdir / "global_eig_hist.png")

    if args.damping is not None and args.damping > 0.0:
        _plot_damping_ratio_hist(layers, args.outdir / "damping_ratio_hist.png")
        _plot_layerwise_medians(layers, args.outdir / "layerwise_balanced_ratios.png")
        _plot_grouped_bars(layers, args.outdir / "grouped_balanced_ratios.png", max_groups=args.max_groups)

    print(f"Wrote outputs to: {args.outdir}")
    print(" - layer_summary.csv")
    print(" - group_summary.csv")
    print(" - global_eig_hist.png")
    if args.damping is not None and args.damping > 0.0:
        print(" - damping_ratio_hist.png")
        print(" - layerwise_balanced_ratios.png")
        print(" - grouped_balanced_ratios.png")


if __name__ == "__main__":
    main()

