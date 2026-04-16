#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected non-empty list of records in {path}")
    return data


def extract_xy(history: list[dict], metric: str) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    for row in history:
        if "step" not in row or metric not in row:
            continue
        x.append(float(row["step"]))
        y.append(float(row[metric]))
    if not x:
        raise ValueError(f"No rows found with both 'step' and '{metric}'")
    return x, y


def skip_first_point(x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    if len(x) <= 1:
        raise ValueError("Cannot skip first point: fewer than 2 points available")
    return x[1:], y[1:]


def maybe_percent(metric: str, values: list[float]) -> list[float]:
    if metric in {"val_acc", "batch_acc", "val_top1", "val_top5", "batch_top1", "batch_top5"}:
        return [v * 100.0 for v in values]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Muon vs KFAC-Muon training curves from JSON logs.")
    parser.add_argument("--muon", type=Path, required=True, help="Path to Muon JSON log")
    parser.add_argument("--kfac", type=Path, required=True, help="Path to KFAC-Muon JSON log")
    parser.add_argument("--metric", type=str, default="val_acc", help="Metric key in JSON records (default: val_acc)")
    parser.add_argument("--title", type=str, default=None, help="Custom plot title")
    parser.add_argument("--out", type=Path, default=Path("compare_training.png"), help="Output image path")
    parser.add_argument("--skip-first", action="store_true", help="Drop the first point from each curve")
    args = parser.parse_args()

    muon_hist = load_history(args.muon)
    kfac_hist = load_history(args.kfac)

    muon_x, muon_y = extract_xy(muon_hist, args.metric)
    kfac_x, kfac_y = extract_xy(kfac_hist, args.metric)

    if args.skip_first:
        muon_x, muon_y = skip_first_point(muon_x, muon_y)
        kfac_x, kfac_y = skip_first_point(kfac_x, kfac_y)

    muon_y = maybe_percent(args.metric, muon_y)
    kfac_y = maybe_percent(args.metric, kfac_y)

    plt.figure(figsize=(8, 5))
    plt.plot(muon_x, muon_y, marker="o", label="Muon")
    plt.plot(kfac_x, kfac_y, marker="o", label="KFAC-Muon")
    plt.xlabel("Step")
    ylabel = f"{args.metric} (%)" if args.metric in {"val_acc", "batch_acc", "val_top1", "val_top5", "batch_top1", "batch_top5"} else args.metric
    plt.ylabel(ylabel)
    plt.title(args.title or f"Muon vs KFAC-Muon: {args.metric} by step")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=180)
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
