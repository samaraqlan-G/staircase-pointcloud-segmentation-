"""Configuration objects for the staircase point-cloud segmentation pipeline.

All geometry is expressed in metres. The label scheme is shared across the
data generator, the segmentation methods and the evaluation utilities so that
predictions and ground truth are always comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- Semantic label scheme (single source of truth) -------------------------
STEPPABLE: int = 0   # horizontal tread surface a foot/wheel can land on
RISER: int = 1       # vertical face between two treads
OTHER: int = 2       # edges, outliers and unstructured LiDAR noise

LABEL_NAMES: dict[int, str] = {STEPPABLE: "steppable", RISER: "riser", OTHER: "other"}
LABEL_COLORS: dict[int, tuple[float, float, float]] = {
    STEPPABLE: (0.15, 0.70, 0.25),  # green
    RISER: (0.85, 0.18, 0.18),      # red
    OTHER: (0.55, 0.55, 0.55),      # grey
}


@dataclass(frozen=True)
class StaircaseConfig:
    """Parametric description of a 4-step staircase and its LiDAR noise model."""

    n_steps: int = 4
    width: float = 2.5            # x-axis extent (2.0 - 3.0 m)
    depth: float = 0.30           # tread depth per step (0.25 - 0.35 m)
    height: float = 0.18          # rise per step (0.15 - 0.20 m)

    # Point budget (total target >= 15_000 per the brief).
    points_per_tread: int = 3000
    points_per_riser: int = 1800

    # Noise model -----------------------------------------------------------
    tread_z_sigma: float = 0.02   # vertical jitter on treads (from the brief)
    riser_y_sigma: float = 0.02   # depth jitter on risers
    lidar_sigma: float = 0.006    # isotropic ranging noise applied to every point
    outlier_ratio: float = 0.03   # fraction of total points injected as outliers
    edge_ratio: float = 0.02      # fraction added as structured edge clutter

    seed: int = 42

    @property
    def total_rise(self) -> float:
        return self.n_steps * self.height

    @property
    def total_run(self) -> float:
        return self.n_steps * self.depth


@dataclass
class SegmentationParams:
    """Hyper-parameters shared by the geometric segmentation methods."""

    # Per-point / per-cluster normal classification thresholds (degrees from +z).
    horizontal_max_angle: float = 30.0   # |angle(n, z)| <= this  -> horizontal
    vertical_min_angle: float = 60.0     # |angle(n, z)| >= this  -> vertical

    normal_k: int = 50                   # k-NN used for local normal estimation

    # RANSAC plane fitting. Band ~2*sigma_z captures a full tread in one plane;
    # the high min-inlier count rejects thin horizontal slices through risers.
    # RANSAC uses its OWN tight orientation tolerance so it never accepts a
    # plane near the oblique staircase incline (~31 deg from horizontal).
    ransac_threshold: float = 0.04       # inlier distance (m)
    ransac_iters: int = 600
    ransac_max_planes: int = 12
    ransac_min_inliers: int = 1000
    ransac_horizontal_max_angle: float = 15.0
    ransac_vertical_min_angle: float = 75.0

    # DBSCAN (normal-augmented feature space: [x, y, z, w*nx, w*ny, w*nz]).
    # Uses a LARGER normal neighbourhood than the per-point method: smoother
    # normals make perpendicular surfaces separate cleanly in feature space,
    # while a high weight w keeps treads and risers in distinct clusters.
    dbscan_eps: float = 0.20
    dbscan_min_samples: int = 20
    dbscan_normal_weight: float = 2.0    # scales normals into the clustering space
    dbscan_normal_k: int = 80            # k-NN for DBSCAN's (smoothed) normals

    # Height-histogram analysis.
    hist_bin: float = 0.01               # z-histogram resolution (m)
    hist_band: float = 0.04              # half-width of a tread band around a peak

    rng_seed: int = 0
