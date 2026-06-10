"""Staircase point-cloud segmentation package."""
from .config import (
    LABEL_COLORS,
    LABEL_NAMES,
    OTHER,
    RISER,
    STEPPABLE,
    SegmentationParams,
    StaircaseConfig,
)
from .data_generation import generate_staircase
from .evaluation import (
    Metrics,
    confusion_matrix,
    evaluate_methods,
    noise_sweep,
    steppable_metrics,
)
from .segmentation import (
    METHODS,
    estimate_normals,
    segment_by_normals,
    segment_dbscan_slope,
    segment_height_histogram,
    segment_ransac,
)

__all__ = [
    "StaircaseConfig", "SegmentationParams",
    "STEPPABLE", "RISER", "OTHER", "LABEL_NAMES", "LABEL_COLORS",
    "generate_staircase",
    "estimate_normals", "segment_by_normals", "segment_ransac",
    "segment_dbscan_slope", "segment_height_histogram", "METHODS",
    "Metrics", "steppable_metrics", "confusion_matrix",
    "evaluate_methods", "noise_sweep",
]
