#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_summary_csv(path: Path) -> dict[str, list[float]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in: {path}")

    data: dict[str, list[float]] = {}
    for key in reader.fieldnames or []:
        values = []
        for row in rows:
            v = _to_float(row.get(key, ""))
            if v is not None:
                values.append(v)
            else:
                values.append(float("nan"))
        data[key] = values
    return data


def choose_x_axis(data: dict[str, list[float]]) -> tuple[str, list[float]]:
    if "epoch" in data:
        return "epoch", data["epoch"]
    # fallback to index
    n = len(next(iter(data.values())))
    return "index", list(range(n))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two timm summary.csv runs.")
    parser.add_argument("--a", type=Path, required=True, help="Path to first summary.csv")
    parser.add_argument("--b", type=Path, required=True, help="Path to second summary.csv")
    parser.add_argument("--label-a", type=str, default="run_a", help="Legend label for --a")
    parser.add_argument("--label-b", type=str, default="run_b", help="Legend label for --b")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["eval_top1", "train_loss"],
        help="Metric columns to plot (default: eval_top1 train_loss)",
    )
    parser.add_argument("--title", type=str, default="Run Comparison", help="Figure title")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("compare_summary.png"),
        help="Output image file",
    )
    args = parser.parse_args()

    data_a = load_summary_csv(args.a)
    data_b = load_summary_csv(args.b)

    x_name_a, x_a = choose_x_axis(data_a)
    x_name_b, x_b = choose_x_axis(data_b)
    x_name = "epoch" if x_name_a == "epoch" and x_name_b == "epoch" else "index"

    metrics = [m for m in args.metrics if m in data_a and m in data_b]
    if not metrics:
        raise ValueError(
            f"No shared metrics found among requested metrics. "
            f"Requested={args.metrics}, shared={set(data_a).intersection(set(data_b))}"
        )

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3.5 * len(metrics)), squeeze=False)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(x_a, data_a[metric], label=args.label_a, linewidth=2)
        ax.plot(x_b, data_b[metric], label=args.label_b, linewidth=2)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend()
        if i == len(metrics) - 1:
            ax.set_xlabel(x_name)

    fig.suptitle(args.title)
    fig.tight_layout(rect=[0, 0.0, 1, 0.98])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

