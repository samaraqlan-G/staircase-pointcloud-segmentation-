"""Geometric segmentation of staircase point clouds.

Four self-contained methods are provided, each mapping a point cloud to a
per-point label in {STEPPABLE, RISER, OTHER}:

1. ``segment_by_normals``   - per-point local-PCA normals + orientation rules.
2. ``segment_ransac``       - sequential multi-plane RANSAC + plane orientation.
3. ``segment_dbscan_slope`` - DBSCAN clustering + per-cluster slope analysis.
4. ``segment_height_histogram`` - z-histogram tread-level detection + normals.

All methods depend only on NumPy and scikit-learn so the notebook runs in any
environment; Open3D is used purely for optional interactive visualisation.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .config import OTHER, RISER, STEPPABLE, SegmentationParams


# --------------------------------------------------------------------------- #
# Shared geometry helpers                                                     #
# --------------------------------------------------------------------------- #
def estimate_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    """Estimate unit normals via PCA over each point's k-nearest neighbourhood.

    The normal is the eigenvector of the local covariance matrix associated
    with the smallest eigenvalue. Normals are sign-disambiguated towards +z so
    that ``|n_z|`` is directly comparable across points.
    """
    n = len(points)
    k = min(k, n)
    nn = NearestNeighbors(n_neighbors=k).fit(points)
    _, idx = nn.kneighbors(points)

    neighbours = points[idx]                                   # (n, k, 3)
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k    # (n, 3, 3)

    eigvals, eigvecs = np.linalg.eigh(cov)                     # ascending eigvals
    normals = eigvecs[:, :, 0]                                 # smallest-eigval vector

    # Disambiguate sign so the z-component is non-negative.
    flip = normals[:, 2] < 0
    normals[flip] *= -1.0
    return normals


def _classify_by_normal_z(nz_abs: np.ndarray, params: SegmentationParams) -> np.ndarray:
    """Map |cos(angle between normal and z)| to a semantic label."""
    cos_h = np.cos(np.deg2rad(params.horizontal_max_angle))
    cos_v = np.cos(np.deg2rad(params.vertical_min_angle))
    labels = np.full(len(nz_abs), OTHER, dtype=np.int64)
    labels[nz_abs >= cos_h] = STEPPABLE
    labels[nz_abs <= cos_v] = RISER
    return labels


def _plane_from_points(p: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane (unit normal, offset d) through >=3 points: n·x + d = 0.

    Uses the eigendecomposition of the 3x3 covariance (O(n) memory) rather than
    a full SVD, whose left-singular matrix would be (n, n) for large clusters.
    """
    centroid = p.mean(axis=0)
    centered = p - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]               # smallest-eigenvalue direction
    d = -normal @ centroid
    return normal, d


# --------------------------------------------------------------------------- #
# Method 1 - per-point normal / PCA analysis                                  #
# --------------------------------------------------------------------------- #
def segment_by_normals(
    points: np.ndarray, params: SegmentationParams | None = None
) -> np.ndarray:
    params = params or SegmentationParams()
    normals = estimate_normals(points, params.normal_k)
    return _classify_by_normal_z(np.abs(normals[:, 2]), params)


# --------------------------------------------------------------------------- #
# Method 2 - sequential multi-plane RANSAC                                     #
# --------------------------------------------------------------------------- #
def _ransac_single_plane(
    points: np.ndarray,
    params: SegmentationParams,
    rng: np.random.Generator,
    horizontal: bool,
    cos_h: float,
    cos_v: float,
) -> np.ndarray:
    """Return a boolean inlier mask for the dominant plane of the requested
    orientation. Candidate planes of the wrong orientation are rejected during
    scoring, so RANSAC never locks onto the oblique staircase-incline plane.
    """
    n = len(points)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    for _ in range(params.ransac_iters):
        sample = points[rng.choice(n, 3, replace=False)]
        normal, d = _plane_from_points(sample)
        if not np.isfinite(normal).all():
            continue
        nz = abs(normal[2])
        ok = (nz >= cos_h) if horizontal else (nz <= cos_v)
        if not ok:
            continue
        dist = np.abs(points @ normal + d)
        inliers = dist < params.ransac_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    return best_inliers


def _extract_planes(
    points: np.ndarray,
    remaining: np.ndarray,
    labels: np.ndarray,
    target_label: int,
    horizontal: bool,
    params: SegmentationParams,
    rng: np.random.Generator,
) -> None:
    """Greedily extract axis-aligned planes of one orientation, in place.

    ``horizontal=True`` accepts only near-horizontal candidates (treads);
    ``horizontal=False`` accepts only near-vertical candidates (risers).
    """
    cos_h = np.cos(np.deg2rad(params.ransac_horizontal_max_angle))
    cos_v = np.cos(np.deg2rad(params.ransac_vertical_min_angle))
    for _ in range(params.ransac_max_planes):
        idx = np.flatnonzero(remaining)
        if len(idx) < params.ransac_min_inliers:
            break
        mask = _ransac_single_plane(points[idx], params, rng, horizontal, cos_h, cos_v)
        if int(mask.sum()) < params.ransac_min_inliers:
            break
        plane_idx = idx[mask]
        labels[plane_idx] = target_label
        remaining[plane_idx] = False


def segment_ransac(
    points: np.ndarray, params: SegmentationParams | None = None
) -> np.ndarray:
    """Two-phase multi-plane RANSAC.

    Risers (vertical planes) are extracted first and removed, because each
    riser's top edge is coplanar in z with the tread above it; extracting
    treads first would otherwise sweep those riser points into the steppable
    class. Remaining structure is then segmented into horizontal tread planes.
    """
    params = params or SegmentationParams()
    rng = np.random.default_rng(params.rng_seed)

    labels = np.full(len(points), OTHER, dtype=np.int64)
    remaining = np.ones(len(points), dtype=bool)

    _extract_planes(points, remaining, labels, RISER, horizontal=False, params=params, rng=rng)
    _extract_planes(points, remaining, labels, STEPPABLE, horizontal=True, params=params, rng=rng)
    return labels


# --------------------------------------------------------------------------- #
# Method 3 - DBSCAN clustering + per-cluster slope analysis                    #
# --------------------------------------------------------------------------- #
def segment_dbscan_slope(
    points: np.ndarray, params: SegmentationParams | None = None
) -> np.ndarray:
    """Normal-aware DBSCAN, then per-cluster slope classification.

    Raw-xyz DBSCAN merges the whole staircase into one connected component
    because adjacent treads and risers touch. We therefore cluster in the
    augmented space ``[x, y, z, w*n]`` so that two perpendicular surfaces that
    are spatially adjacent are still separated by their differing normals.
    """
    params = params or SegmentationParams()
    normals = estimate_normals(points, params.dbscan_normal_k)
    features = np.hstack([points, params.dbscan_normal_weight * normals])

    db = DBSCAN(
        eps=params.dbscan_eps,
        min_samples=params.dbscan_min_samples,
        algorithm="kd_tree",          # explicit: avoid the 'auto'->brute O(n^2) blow-up
        n_jobs=1,
    )
    cluster_ids = db.fit_predict(features)

    labels = np.full(len(points), OTHER, dtype=np.int64)
    cos_h = np.cos(np.deg2rad(params.horizontal_max_angle))
    cos_v = np.cos(np.deg2rad(params.vertical_min_angle))

    for cid in np.unique(cluster_ids):
        if cid == -1:                       # DBSCAN noise stays OTHER
            continue
        mask = cluster_ids == cid
        if mask.sum() < 3:
            continue
        normal, _ = _plane_from_points(points[mask])
        nz = abs(normal[2])
        if nz >= cos_h:
            labels[mask] = STEPPABLE
        elif nz <= cos_v:
            labels[mask] = RISER
        # else: leave as OTHER (oblique / mixed cluster)
    return labels


# --------------------------------------------------------------------------- #
# Method 4 - height-histogram tread detection (advanced)                       #
# --------------------------------------------------------------------------- #
def _detect_z_peaks(z: np.ndarray, params: SegmentationParams) -> np.ndarray:
    edges = np.arange(z.min(), z.max() + params.hist_bin, params.hist_bin)
    counts, edges = np.histogram(z, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(counts) < 3:
        return centers[np.argmax(counts)][None] if len(counts) else np.array([])
    # Local maxima above an adaptive prominence threshold.
    thr = 0.25 * counts.max()
    is_peak = (
        (counts[1:-1] >= counts[:-2])
        & (counts[1:-1] >= counts[2:])
        & (counts[1:-1] >= thr)
    )
    peaks = centers[1:-1][is_peak]
    return peaks


def segment_height_histogram(
    points: np.ndarray, params: SegmentationParams | None = None
) -> np.ndarray:
    """Detect discrete tread levels from the z-histogram, then refine with normals."""
    params = params or SegmentationParams()
    z = points[:, 2]
    peaks = _detect_z_peaks(z, params)

    labels = np.full(len(points), OTHER, dtype=np.int64)
    if peaks.size:
        near_peak = np.min(np.abs(z[:, None] - peaks[None, :]), axis=1) <= params.hist_band
    else:
        near_peak = np.zeros(len(points), dtype=bool)

    # A point is steppable only if it sits in a tread band *and* is locally flat.
    normals = estimate_normals(points, params.normal_k)
    nz = np.abs(normals[:, 2])
    cos_h = np.cos(np.deg2rad(params.horizontal_max_angle))
    cos_v = np.cos(np.deg2rad(params.vertical_min_angle))

    labels[near_peak & (nz >= cos_h)] = STEPPABLE
    labels[(~near_peak) & (nz <= cos_v)] = RISER
    return labels


METHODS = {
    "normals": segment_by_normals,
    "ransac": segment_ransac,
    "dbscan_slope": segment_dbscan_slope,
    "height_histogram": segment_height_histogram,
}
