"""Visualisation helpers (matplotlib for headless PNG export, Open3D optional)."""
from __future__ import annotations

import numpy as np

from .config import LABEL_COLORS, LABEL_NAMES


def _colors_for(labels: np.ndarray) -> np.ndarray:
    c = np.zeros((len(labels), 3))
    for k, col in LABEL_COLORS.items():
        c[labels == k] = col
    return c


def plot_pointcloud_mpl(
    points: np.ndarray,
    labels: np.ndarray,
    title: str = "",
    ax=None,
    elev: float = 22.0,
    azim: float = -60.0,
    s: float = 1.5,
):
    """3D scatter coloured by label. Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt  # local import keeps module import light

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=_colors_for(labels), s=s, depthshade=True, linewidths=0,
    )
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect(
            (np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2]))
        )
    except Exception:
        pass
    return ax


def save_comparison_figure(
    points: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str,
    pred_title: str = "Prediction",
):
    """Side-by-side ground-truth vs prediction figure with a shared legend."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    plot_pointcloud_mpl(points, y_true, "Ground truth", ax=ax1)
    plot_pointcloud_mpl(points, y_pred, pred_title, ax=ax2)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=col,
               markeredgecolor=col, label=LABEL_NAMES[k])
        for k, col in LABEL_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def show_open3d(points: np.ndarray, labels: np.ndarray):  # pragma: no cover
    """Interactive Open3D viewer (optional dependency)."""
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install open3d for interactive visualisation: pip install open3d") from exc

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(_colors_for(labels))
    o3d.visualization.draw_geometries([pc])
