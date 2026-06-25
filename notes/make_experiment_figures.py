#!/usr/bin/env python3
"""Generate lightweight vector figures for the KFAC-Muon experiment note."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "notes" / "figures"
TABLE_DIR = ROOT / "notes" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "imagenet_muon": Path("/Users/luke/Downloads/muon_b256_e90/summary.csv"),
    "imagenet_kfac": Path("/Users/luke/Downloads/kfac_b256_e90/summary.csv"),
    "cifar_muon_1e3": Path("/Users/luke/Downloads/vits16_c100_last_sweep/summaries/vits16_c100_muon_lr1e3_e75_s12/summary.csv"),
    "cifar_muon_1p2e3": Path("/Users/luke/Downloads/vits16_c100_last_sweep/summaries/vits16_c100_muon_lr1p2e3_e75_s12/summary.csv"),
    "cifar_kfac": Path("/Users/luke/Downloads/vits16_c100_partial/summaries/vits16_c100_kfac_lmoff_damp5e5_inclfl_e75_s12/summary.csv"),
    "cifar_muon_200": Path("/Users/luke/Downloads/kfac_runs/vits16_cifar100_muon_e200_baseline/summary.csv"),
    "cifar_kfac_200": Path("/Users/luke/Downloads/kfac_runs/vits16_cifar100_kfacmuon_e200_adapt_baseline/summary.csv"),
}

BLUE = colors.HexColor("#2860A8")
ORANGE = colors.HexColor("#D87024")
GREEN = colors.HexColor("#2C8C4A")
GRAY = colors.HexColor("#7A7A7A")
DARK = colors.HexColor("#202020")
LIGHT_GRID = colors.HexColor("#DDDDDD")

@dataclass
class Series:
    name: str
    xs: list[float]
    ys: list[float]
    color: colors.Color
    dash: tuple[int, int] | None = None


def read_summary(path: Path) -> list[dict[str, float]]:
    rows_by_epoch: dict[int, dict[str, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if not raw.get("epoch") or raw.get("epoch") == "epoch":
                continue
            row: dict[str, float] = {}
            for key, val in raw.items():
                if not key or val is None or val == "":
                    continue
                try:
                    row[key] = float(val) if str(val).lower() != "nan" else math.nan
                except ValueError:
                    pass
            if "epoch" in row:
                rows_by_epoch[int(row["epoch"])] = row
    return [rows_by_epoch[e] for e in sorted(rows_by_epoch)]


def metric(rows: list[dict[str, float]], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for row in rows:
        if key in row and not math.isnan(row[key]):
            xs.append(row["epoch"])
            ys.append(row[key])
    return xs, ys


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, n - 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min([1, 2, 2.5, 5, 10], key=lambda x: abs(raw - x * mag)) * mag
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 0.5 * step:
        if v >= lo - 1e-9:
            ticks.append(v)
        v += step
    return ticks[:8]


def draw_plot(c: canvas.Canvas, x: float, y: float, w: float, h: float, series: list[Series],
              title: str | None, xlabel: str, ylabel: str, y_pad: float = 0.04,
              y_bounds: tuple[float, float] | None = None,
              hlines: list[tuple[float, colors.Color, tuple[int, int] | None]] | None = None):
    if title:
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(DARK)
        c.drawString(x + 46, y + h - 2, title)

    all_x = [v for s in series for v in s.xs]
    all_y = [v for s in series for v in s.ys]
    xmin, xmax = min(all_x), max(all_x)
    if y_bounds is None:
        ymin, ymax = min(all_y), max(all_y)
        pad = (ymax - ymin) * y_pad if ymax > ymin else 1.0
        ymin -= pad
        ymax += pad
    else:
        ymin, ymax = y_bounds

    left_pad, right_pad, bottom_pad, top_pad = 46, 12, 34, 14
    px, py = x + left_pad, y + bottom_pad
    pw, ph = w - left_pad - right_pad, h - bottom_pad - top_pad

    def sx(v: float) -> float:
        return px + (v - xmin) / (xmax - xmin) * pw if xmax > xmin else px

    def sy(v: float) -> float:
        return py + (v - ymin) / (ymax - ymin) * ph if ymax > ymin else py

    # Grid and axes.
    c.setStrokeColor(LIGHT_GRID)
    c.setLineWidth(0.35)
    yticks = nice_ticks(ymin, ymax, 5)
    for t in yticks:
        c.line(px, sy(t), px + pw, sy(t))
    xticks = nice_ticks(xmin, xmax, 6)
    for t in xticks:
        c.line(sx(t), py, sx(t), py + ph)

    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)
    c.line(px, py, px + pw, py)
    c.line(px, py, px, py + ph)

    if hlines:
        for value, color, dash in hlines:
            if ymin <= value <= ymax:
                c.setStrokeColor(color)
                c.setLineWidth(0.9)
                if dash:
                    c.setDash(*dash)
                else:
                    c.setDash()
                c.line(px, sy(value), px + pw, sy(value))
        c.setDash()

    c.setFont("Helvetica", 8)
    c.setFillColor(DARK)
    for t in yticks:
        label = f"{t:.0f}" if abs(t) >= 10 else f"{t:.2f}"
        c.drawRightString(px - 5, sy(t) - 3, label)
    for t in xticks:
        c.drawCentredString(sx(t), py - 14, f"{int(round(t))}")

    c.setFont("Helvetica", 9)
    c.drawCentredString(px + pw / 2, y + 2, xlabel)
    c.saveState()
    c.translate(x + 8, py + ph / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()

    # Lines.
    for s in series:
        c.setStrokeColor(s.color)
        c.setLineWidth(1.8)
        if s.dash:
            c.setDash(*s.dash)
        else:
            c.setDash()
        points = list(zip(s.xs, s.ys))
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            c.line(sx(x0), sy(y0), sx(x1), sy(y1))
        # sparse markers
        stride = max(1, len(points) // 10)
        c.setFillColor(colors.white)
        for i, (xv, yv) in enumerate(points):
            if i % stride == 0 or i == len(points) - 1:
                c.circle(sx(xv), sy(yv), 2.0, stroke=1, fill=1)
    c.setDash()

    # Legend.
    lx, ly = x + w - 112, y + h - 2
    c.setFont("Helvetica", 8.5)
    legend_items = [s for s in series if s.name]
    for i, s in enumerate(legend_items):
        yy = ly - 13 * i
        c.setStrokeColor(s.color)
        c.setLineWidth(2)
        if s.dash:
            c.setDash(*s.dash)
        else:
            c.setDash()
        c.line(lx, yy + 3, lx + 20, yy + 3)
        c.setDash()
        c.setFillColor(DARK)
        c.drawString(lx + 25, yy, s.name)


def draw_two_panel_pdf(path: Path, left_series: list[Series], right_series: list[Series], title: str,
                       left_title: str, right_title: str, left_ylabel: str, right_ylabel: str,
                       left_y_bounds: tuple[float, float] | None = None,
                       right_y_bounds: tuple[float, float] | None = None,
                       right_hlines: list[tuple[float, colors.Color, tuple[int, int] | None]] | None = None):
    width, height = letter[0], 330
    c = canvas.Canvas(str(path), pagesize=(width, height))
    c.setTitle(title)
    draw_plot(c, 36, 42, 258, 245, left_series, left_title, "epoch", left_ylabel, y_bounds=left_y_bounds)
    draw_plot(c, 318, 42, 258, 245, right_series, right_title, "epoch", right_ylabel,
              y_bounds=right_y_bounds, hlines=right_hlines)
    c.save()


def draw_bar_pdf(path: Path, data: list[tuple[str, float, float]], title: str):
    width, height = 430, 280
    c = canvas.Canvas(str(path), pagesize=(width, height))
    c.setTitle(title)
    x0, y0, w, h = 58, 50, 320, 190
    ymin = math.floor((min(min(a,b) for _, a, b in data)-2)/5)*5
    ymax = math.ceil((max(max(a,b) for _, a, b in data)+2)/5)*5
    def sy(v): return y0 + (v-ymin)/(ymax-ymin)*h
    c.setStrokeColor(LIGHT_GRID)
    c.setLineWidth(0.35)
    for t in nice_ticks(ymin,ymax,6):
        c.line(x0, sy(t), x0+w, sy(t))
        c.setFont("Helvetica",8)
        c.setFillColor(DARK)
        c.drawRightString(x0-6, sy(t)-3, f"{t:.0f}")
    c.setStrokeColor(colors.black); c.setLineWidth(0.8)
    c.line(x0,y0,x0+w,y0); c.line(x0,y0,x0,y0+h)
    group_w = w/len(data)
    bar_w = 34
    c.setFont("Helvetica",8.5)
    for i,(label,muon,kfac) in enumerate(data):
        gx = x0 + i*group_w + group_w/2
        for j,(val,col,name) in enumerate([(muon,BLUE,"Muon"),(kfac,ORANGE,"KFAC-Muon")]):
            bx = gx + (j-0.5)*(bar_w+4)
            c.setFillColor(col)
            c.rect(bx, y0, bar_w, sy(val)-y0, stroke=0, fill=1)
            c.setFillColor(DARK)
            c.drawCentredString(bx+bar_w/2, sy(val)+4, f"{val:.2f}")
        c.drawCentredString(gx, y0-18, label)
    c.setFont("Helvetica",9)
    c.drawCentredString(x0+w/2, 18, "best top-1 (%)")
    # Legend
    c.setFillColor(BLUE); c.rect(width-145,height-54,10,10,stroke=0,fill=1)
    c.setFillColor(DARK); c.drawString(width-130,height-53,"Muon")
    c.setFillColor(ORANGE); c.rect(width-145,height-70,10,10,stroke=0,fill=1)
    c.setFillColor(DARK); c.drawString(width-130,height-69,"KFAC")
    c.save()


def draw_loss_pdf(path: Path, cifar_series: list[Series], imagenet_series: list[Series]):
    width, height = letter[0], 330
    c = canvas.Canvas(str(path), pagesize=(width, height))
    c.setTitle("Validation loss comparisons")
    draw_plot(c, 36, 42, 258, 245, cifar_series, "CIFAR-100", "epoch", "loss",
              y_bounds=(1.0, 2.4))
    draw_plot(c, 318, 42, 258, 245, imagenet_series, "ImageNet-100", "epoch", "loss",
              y_bounds=(0.75, 2.25))
    c.save()


def best(rows: list[dict[str,float]]) -> dict[str,float]:
    return max(rows, key=lambda r: r["eval_top1"])


def final(rows: list[dict[str,float]]) -> dict[str,float]:
    return rows[-1]


def trim(xs: list[float], ys: list[float], start_epoch: int = 0) -> tuple[list[float], list[float]]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x >= start_epoch]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def aligned_gap(rows_a: list[dict[str, float]], rows_b: list[dict[str, float]],
                key: str = "eval_top1", start_epoch: int = 0) -> tuple[list[float], list[float]]:
    by_a = {int(r["epoch"]): r for r in rows_a}
    by_b = {int(r["epoch"]): r for r in rows_b}
    xs, ys = [], []
    for epoch in sorted(set(by_a) & set(by_b)):
        if epoch < start_epoch:
            continue
        va = by_a[epoch].get(key, math.nan)
        vb = by_b[epoch].get(key, math.nan)
        if not math.isnan(va) and not math.isnan(vb):
            xs.append(float(epoch))
            ys.append(va - vb)
    return xs, ys


def write_results_table(rows_by_name: dict[str, list[dict[str,float]]]):
    rows = []
    configs = [
        ("CIFAR-100", "ViT-S/16", "Muon", rows_by_name["cifar_muon_1p2e3"]),
        ("CIFAR-100", "ViT-S/16", "KFAC-Muon", rows_by_name["cifar_kfac"]),
        ("ImageNet-100", "ViT-B/16", "Muon", rows_by_name["imagenet_muon"]),
        ("ImageNet-100", "ViT-B/16", "KFAC-Muon", rows_by_name["imagenet_kfac"]),
    ]
    with (TABLE_DIR / "main_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model", "optimizer", "epochs", "best_epoch", "best_top1", "final_top1", "best_loss", "final_loss"])
        for dataset, model, opt, rs in configs:
            b, l = best(rs), final(rs)
            writer.writerow([dataset, model, opt, len(rs), int(b["epoch"]), f"{b['eval_top1']:.4f}", f"{l['eval_top1']:.4f}", f"{b['eval_loss']:.6f}", f"{l['eval_loss']:.6f}"])


def main():
    rows = {name: read_summary(path) for name, path in RUNS.items() if path.exists()}

    # ImageNet-100 figure.
    im_muon_top_x, im_muon_top_y = trim(*metric(rows["imagenet_muon"], "eval_top1"), start_epoch=5)
    im_kfac_top_x, im_kfac_top_y = trim(*metric(rows["imagenet_kfac"], "eval_top1"), start_epoch=5)
    im_gap_x, im_gap_y = aligned_gap(rows["imagenet_kfac"], rows["imagenet_muon"], start_epoch=5)
    draw_two_panel_pdf(
        FIG_DIR / "imagenet100_vitb16_b256_curves.pdf",
        [Series("Muon", im_muon_top_x, im_muon_top_y, BLUE), Series("KFAC", im_kfac_top_x, im_kfac_top_y, ORANGE)],
        [Series("", im_gap_x, im_gap_y, ORANGE)],
        "ImageNet-100, ViT-B/16, batch 256",
        "top-1", "gap", "top-1 (%)", "points",
        left_y_bounds=(44, 85), right_y_bounds=(-0.5, 6.5),
        right_hlines=[(0.0, GRAY, (3, 3))],
    )

    # CIFAR-100 figure.
    cm_top_x, cm_top_y = trim(*metric(rows["cifar_muon_1p2e3"], "eval_top1"), start_epoch=5)
    ck_top_x, ck_top_y = trim(*metric(rows["cifar_kfac"], "eval_top1"), start_epoch=5)
    cg_x, cg_y = aligned_gap(rows["cifar_kfac"], rows["cifar_muon_1p2e3"], start_epoch=5)
    draw_two_panel_pdf(
        FIG_DIR / "cifar100_vits16_curves.pdf",
        [Series("Muon", cm_top_x, cm_top_y, BLUE), Series("KFAC", ck_top_x, ck_top_y, ORANGE)],
        [Series("", cg_x, cg_y, ORANGE)],
        "CIFAR-100, ViT-S/16, batch 128",
        "top-1", "gap", "top-1 (%)", "points",
        left_y_bounds=(38, 76.5), right_y_bounds=(-0.5, 4.5),
        right_hlines=[(0.0, GRAY, (3, 3))],
    )

    # Validation-loss comparison figure.
    cm_loss_x, cm_loss_y = trim(*metric(rows["cifar_muon_1p2e3"], "eval_loss"), start_epoch=5)
    ck_loss_x, ck_loss_y = trim(*metric(rows["cifar_kfac"], "eval_loss"), start_epoch=5)
    im_muon_loss_x, im_muon_loss_y = trim(*metric(rows["imagenet_muon"], "eval_loss"), start_epoch=5)
    im_kfac_loss_x, im_kfac_loss_y = trim(*metric(rows["imagenet_kfac"], "eval_loss"), start_epoch=5)
    draw_loss_pdf(
        FIG_DIR / "validation_loss_curves.pdf",
        [Series("Muon", cm_loss_x, cm_loss_y, BLUE), Series("KFAC", ck_loss_x, ck_loss_y, ORANGE)],
        [Series("Muon", im_muon_loss_x, im_muon_loss_y, BLUE), Series("KFAC", im_kfac_loss_x, im_kfac_loss_y, ORANGE)],
    )

    draw_bar_pdf(
        FIG_DIR / "main_results_best_top1.pdf",
        [
            ("CIFAR-100", best(rows["cifar_muon_1p2e3"])["eval_top1"], best(rows["cifar_kfac"])["eval_top1"]),
            ("ImageNet-100", best(rows["imagenet_muon"])["eval_top1"], best(rows["imagenet_kfac"])["eval_top1"]),
        ],
        "Best validation top-1",
    )
    write_results_table(rows)

    print("wrote:")
    for p in sorted(FIG_DIR.glob("*.pdf")):
        print(p)
    print(TABLE_DIR / "main_results.csv")

if __name__ == "__main__":
    main()
