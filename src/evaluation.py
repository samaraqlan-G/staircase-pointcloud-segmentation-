"""Performance evaluation for staircase segmentation.

Metrics follow the brief: overall accuracy plus precision, recall, F1 and IoU
for the safety-critical *steppable* class. A small noise-sensitivity sweep is
included to quantify robustness against LiDAR noise and outliers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import OTHER, RISER, STEPPABLE, StaircaseConfig
from .data_generation import generate_staircase


@dataclass
class Metrics:
    accuracy: float
    precision: float        # steppable class
    recall: float           # steppable class
    f1: float               # steppable class
    iou: float              # steppable class

    def as_row(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def steppable_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Binary metrics for the steppable class, plus multi-class accuracy."""
    accuracy = float((y_true == y_pred).mean())

    gt_pos = y_true == STEPPABLE
    pr_pos = y_pred == STEPPABLE
    tp = float((gt_pos & pr_pos).sum())
    fp = float((~gt_pos & pr_pos).sum())
    fn = float((gt_pos & ~pr_pos).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return Metrics(accuracy, precision, recall, f1, iou)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """3x3 confusion matrix ordered [STEPPABLE, RISER, OTHER]."""
    classes = [STEPPABLE, RISER, OTHER]
    cm = np.zeros((3, 3), dtype=np.int64)
    for i, ct in enumerate(classes):
        for j, cp in enumerate(classes):
            cm[i, j] = int(((y_true == ct) & (y_pred == cp)).sum())
    return cm


def evaluate_methods(
    points: np.ndarray, y_true: np.ndarray, methods: dict, params=None
) -> dict[str, Metrics]:
    out: dict[str, Metrics] = {}
    for name, fn in methods.items():
        y_pred = fn(points, params) if params is not None else fn(points)
        out[name] = steppable_metrics(y_true, y_pred)
    return out


def noise_sweep(
    method,
    sigmas: list[float],
    outlier_ratios: list[float],
    base_cfg: StaircaseConfig | None = None,
    params=None,
) -> dict[str, list[tuple[float, float]]]:
    """Sweep LiDAR sigma and outlier ratio independently; report (level, F1).

    Returns a dict with keys ``"lidar_sigma"`` and ``"outlier_ratio"`` mapping to
    lists of ``(level, steppable_f1)`` tuples.
    """
    base_cfg = base_cfg or StaircaseConfig()
    result: dict[str, list[tuple[float, float]]] = {"lidar_sigma": [], "outlier_ratio": []}

    for s in sigmas:
        pts, lab = generate_staircase(base_cfg, lidar_sigma=s)
        pred = method(pts, params) if params is not None else method(pts)
        result["lidar_sigma"].append((s, steppable_metrics(lab, pred).f1))

    for o in outlier_ratios:
        pts, lab = generate_staircase(base_cfg, outlier_ratio=o)
        pred = method(pts, params) if params is not None else method(pts)
        result["outlier_ratio"].append((o, steppable_metrics(lab, pred).f1))

    return result
