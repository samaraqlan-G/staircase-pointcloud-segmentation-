"""Synthetic staircase point-cloud generation.

Implements the mathematical model from the brief. For each step ``i``:

Tread (horizontal, steppable)::

    x ~ U(0, width)
    y ~ U(i * depth, (i + 1) * depth)
    z = i * height + eps_z,        eps_z ~ N(0, tread_z_sigma^2)

Riser (vertical face joining tread ``i-1`` to tread ``i``)::

    x ~ U(0, width)
    y = i * depth + eps_y,         eps_y ~ N(0, riser_y_sigma^2)
    z ~ U((i - 1) * height, i * height)

On top of the structural model we add isotropic LiDAR ranging noise, a band of
structured edge clutter and a population of uniform outliers, each carrying the
correct ground-truth semantic label.
"""
from __future__ import annotations

import numpy as np

from .config import OTHER, RISER, STEPPABLE, StaircaseConfig


def _tread_points(i: int, cfg: StaircaseConfig, rng: np.random.Generator) -> np.ndarray:
    n = cfg.points_per_tread
    x = rng.uniform(0.0, cfg.width, n)
    y = rng.uniform(i * cfg.depth, (i + 1) * cfg.depth, n)
    z = i * cfg.height + rng.normal(0.0, cfg.tread_z_sigma, n)
    return np.column_stack((x, y, z))


def _riser_points(i: int, cfg: StaircaseConfig, rng: np.random.Generator) -> np.ndarray:
    n = cfg.points_per_riser
    x = rng.uniform(0.0, cfg.width, n)
    y = i * cfg.depth + rng.normal(0.0, cfg.riser_y_sigma, n)
    z = rng.uniform((i - 1) * cfg.height, i * cfg.height, n)
    return np.column_stack((x, y, z))


def _edge_points(cfg: StaircaseConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    """Structured clutter along the lateral edges of the staircase (x ~ 0 or width)."""
    side = rng.integers(0, 2, n)
    x = np.where(side == 0, 0.0, cfg.width) + rng.normal(0.0, 0.01, n)
    y = rng.uniform(0.0, cfg.total_run, n)
    z = rng.uniform(0.0, cfg.total_rise, n) + rng.normal(0.0, 0.03, n)
    return np.column_stack((x, y, z))


def _outlier_points(cfg: StaircaseConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform outliers spread through an inflated bounding box (LiDAR ghost returns)."""
    x = rng.uniform(-0.2, cfg.width + 0.2, n)
    y = rng.uniform(-0.2, cfg.total_run + 0.2, n)
    z = rng.uniform(-0.1, cfg.total_rise + 0.25, n)
    return np.column_stack((x, y, z))


def generate_staircase(
    cfg: StaircaseConfig | None = None,
    lidar_sigma: float | None = None,
    outlier_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a labelled staircase point cloud.

    Parameters
    ----------
    cfg:
        Geometry / noise configuration. Defaults to :class:`StaircaseConfig`.
    lidar_sigma, outlier_ratio:
        Optional overrides, convenient for noise-sensitivity sweeps without
        rebuilding a whole config object.

    Returns
    -------
    points : (N, 3) float64 array
    labels : (N,) int array with values in {STEPPABLE, RISER, OTHER}
    """
    cfg = cfg or StaircaseConfig()
    lidar_sigma = cfg.lidar_sigma if lidar_sigma is None else lidar_sigma
    outlier_ratio = cfg.outlier_ratio if outlier_ratio is None else outlier_ratio

    rng = np.random.default_rng(cfg.seed)

    clouds: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    # Treads: i = 0 .. n_steps - 1
    for i in range(cfg.n_steps):
        pts = _tread_points(i, cfg, rng)
        clouds.append(pts)
        labels.append(np.full(len(pts), STEPPABLE, dtype=np.int64))

    # Risers: i = 1 .. n_steps - 1 (vertical faces connecting consecutive treads)
    for i in range(1, cfg.n_steps):
        pts = _riser_points(i, cfg, rng)
        clouds.append(pts)
        labels.append(np.full(len(pts), RISER, dtype=np.int64))

    structural_n = sum(len(c) for c in clouds)

    n_edge = int(cfg.edge_ratio * structural_n)
    if n_edge:
        pts = _edge_points(cfg, n_edge, rng)
        clouds.append(pts)
        labels.append(np.full(len(pts), OTHER, dtype=np.int64))

    n_out = int(outlier_ratio * structural_n)
    if n_out:
        pts = _outlier_points(cfg, n_out, rng)
        clouds.append(pts)
        labels.append(np.full(len(pts), OTHER, dtype=np.int64))

    points = np.vstack(clouds)
    labels_arr = np.concatenate(labels)

    # Isotropic LiDAR ranging noise on every point.
    if lidar_sigma > 0:
        points = points + rng.normal(0.0, lidar_sigma, points.shape)

    # Shuffle so downstream methods cannot exploit point ordering.
    perm = rng.permutation(len(points))
    return points[perm], labels_arr[perm]


if __name__ == "__main__":  # pragma: no cover
    pts, lab = generate_staircase()
    print(f"generated {len(pts)} points")
    for k in (STEPPABLE, RISER, OTHER):
        print(f"  label {k}: {(lab == k).sum()}")
