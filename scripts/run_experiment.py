"""Run the full mini-project experiment and emit figures + a results table.

Outputs (written to ``results/``):
    raw_pointcloud.png          ground-truth coloured cloud
    seg_<method>.png            GT vs prediction for each method
    noise_sensitivity.png       steppable-F1 vs LiDAR sigma / outlier ratio
    confusion_<best>.png        confusion matrix of the best method
    metrics.json                all metrics, ready to paste into the paper
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import (
    METHODS,
    SegmentationParams,
    StaircaseConfig,
    confusion_matrix,
    evaluate_methods,
    generate_staircase,
    noise_sweep,
)
from src.config import LABEL_NAMES
from src.visualization import plot_pointcloud_mpl, save_comparison_figure

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def main() -> None:
    cfg = StaircaseConfig()
    params = SegmentationParams()

    points, y_true = generate_staircase(cfg)
    print(f"[data] {len(points)} points | "
          + " ".join(f"{LABEL_NAMES[k]}={int((y_true==k).sum())}" for k in (0, 1, 2)))

    # Raw / ground-truth cloud.
    ax = plot_pointcloud_mpl(points, y_true, "Ground-truth staircase point cloud")
    ax.figure.savefig(os.path.join(RESULTS, "raw_pointcloud.png"), dpi=160, bbox_inches="tight")
    plt.close(ax.figure)

    # Run + time every method, save comparison figures.
    metrics_table: dict[str, dict] = {}
    timings: dict[str, float] = {}
    for name, fn in METHODS.items():
        t0 = time.perf_counter()
        y_pred = fn(points, params)
        timings[name] = time.perf_counter() - t0
        m = evaluate_methods(points, y_true, {name: fn}, params)[name]
        metrics_table[name] = {**m.as_row(), "runtime_s": round(timings[name], 3)}
        save_comparison_figure(
            points, y_true, y_pred,
            os.path.join(RESULTS, f"seg_{name}.png"),
            pred_title=f"{name} (F1={m.f1:.3f}, IoU={m.iou:.3f})",
        )
        print(f"[seg ] {name:18s} acc={m.accuracy:.3f} P={m.precision:.3f} "
              f"R={m.recall:.3f} F1={m.f1:.3f} IoU={m.iou:.3f} ({timings[name]:.2f}s)")

    best = max(metrics_table, key=lambda k: metrics_table[k]["f1"])
    print(f"[best] {best}")

    # Confusion matrix of the best method.
    y_pred_best = METHODS[best](points, params)
    cm = confusion_matrix(y_true, y_pred_best)
    fig, axc = plt.subplots(figsize=(4.6, 4.2))
    im = axc.imshow(cm, cmap="Blues")
    axc.set_xticks(range(3)); axc.set_yticks(range(3))
    axc.set_xticklabels([LABEL_NAMES[i] for i in range(3)])
    axc.set_yticklabels([LABEL_NAMES[i] for i in range(3)])
    axc.set_xlabel("Predicted"); axc.set_ylabel("True")
    axc.set_title(f"Confusion matrix - {best}")
    for i in range(3):
        for j in range(3):
            axc.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"confusion_{best}.png"), dpi=160)
    plt.close(fig)

    # Noise sensitivity on the best method.
    sigmas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    outliers = [0.0, 0.03, 0.06, 0.1, 0.15, 0.2]
    sweep = noise_sweep(METHODS[best], sigmas, outliers, cfg, params)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    xs, ys = zip(*sweep["lidar_sigma"])
    a1.plot(xs, ys, "o-"); a1.set_xlabel("LiDAR sigma (m)")
    a1.set_ylabel("steppable F1"); a1.set_title("F1 vs ranging noise"); a1.grid(alpha=0.3)
    xs, ys = zip(*sweep["outlier_ratio"])
    a2.plot(xs, ys, "s-", color="C3"); a2.set_xlabel("outlier ratio")
    a2.set_ylabel("steppable F1"); a2.set_title("F1 vs outlier ratio"); a2.grid(alpha=0.3)
    fig.suptitle(f"Noise sensitivity - {best}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "noise_sensitivity.png"), dpi=160)
    plt.close(fig)

    out = {
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "n_points": int(len(points)),
        "metrics": metrics_table,
        "best_method": best,
        "confusion_matrix_best": cm.tolist(),
        "noise_sweep": sweep,
    }
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] results written to {os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
