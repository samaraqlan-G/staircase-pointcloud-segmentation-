# Geometric Segmentation of Steppable Surfaces in Simulated LiDAR Staircase Point Clouds

A comparative evaluation of four classical (non-learning) geometric segmentation
methods for separating **steppable treads** from **risers** and **other** points
in a synthetic 4-step staircase point cloud, framed for **stair-climbing robot
perception**.

This repository accompanies a short IEEE-format paper and a runnable Jupyter
notebook. Everything is dependency-light (NumPy + scikit-learn); Open3D is
optional and only used for interactive 3D viewing.

---

## Problem

A stair-following robot must decide **where it is safe to place a foot**. Given a
LiDAR point cloud of a staircase, we label every point as one of:

| Label | Meaning | Colour |
|-------|---------|--------|
| `STEPPABLE` (0) | horizontal tread — safe to step on | green |
| `RISER` (1) | vertical face between steps | red |
| `OTHER` (2) | edges, noise, outliers | grey |

The dominant safety criterion is **precision of the steppable class**: a false
positive (calling a riser/edge "steppable") can cause a misstep, whereas a false
negative merely discards a usable foothold.

## Data generation

A parametric generator produces the staircase analytically. For step
`i = 0..3`, with `width`, `depth`, `height`:

**Tread (steppable):**
```
x ~ U(0, width)
y ~ U(i*depth, (i+1)*depth)
z = i*height + e_z,    e_z ~ N(0, 0.02^2)
```

**Riser (vertical):**
```
x ~ U(0, width)
y = i*depth + e_y,     e_y ~ N(0, 0.02^2)
z ~ U((i-1)*height, i*height)
```

On top of this we add isotropic LiDAR sensor noise, a fraction of uniform
outliers, and a thin band of edge clutter at each tread/riser join. Default
output: **18,270 points** (12,000 steppable / 5,400 riser / 870 other).

## Methods

| # | Method | Idea |
|---|--------|------|
| 1 | **PCA + Normal analysis** | per-point normal via local-PCA over k-NN; classify by the angle of the normal to the z-axis |
| 2 | **RANSAC multi-plane** | iteratively fit planes; vertical planes first (risers), then horizontal (treads), with tight orientation tolerances |
| 3 | **DBSCAN + slope** | cluster in an augmented `[x,y,z, w*normal]` space, then classify each cluster by its fitted plane normal |
| 4 | **Height-histogram** (advanced) | detect tread elevations as peaks in the z-histogram, assign points within a band around each peak |

## Results

Steppable-class metrics on the default cloud (`seed=42`, 18,270 points):

| Method | Acc | Precision | Recall | F1 | IoU | Runtime |
|--------|-----|-----------|--------|----|----|---------|
| **Normals (PCA)** | **0.794** | 0.966 | **0.865** | **0.913** | **0.840** | 0.29 s |
| Height-histogram | 0.722 | 0.975 | 0.818 | 0.889 | 0.801 | 0.28 s |
| DBSCAN + slope | 0.769 | **0.977** | 0.796 | 0.877 | 0.782 | 0.83 s |
| RANSAC multi-plane | 0.711 | 0.888 | 0.701 | 0.784 | 0.644 | 0.15 s |

**Takeaways**
- The PCA-normal method gives the best F1/IoU and recall.
- DBSCAN and height-histogram reach the **highest steppable precision (~0.98)** —
  the most safety-relevant metric for foot placement.
- Under increasing LiDAR noise, steppable F1 degrades from 0.918 (σ=0) to
  0.577 (σ=0.05); the per-step `σ_z = 0.02` tread jitter mandated by the brief is
  already enough to noticeably corrupt local normals.
- Outliers are handled gracefully (F1 0.918 → 0.875 from 0% to 20% outliers).

See `results/` for figures (`raw_pointcloud.png`, `seg_*.png`,
`confusion_normals.png`, `noise_sensitivity.png`) and `metrics.json`.

## Repository layout

```
staircase-pointcloud-segmentation/
├── src/
│   ├── config.py           # geometry + segmentation hyper-parameters, label scheme
│   ├── data_generation.py  # parametric staircase point-cloud generator
│   ├── segmentation.py     # the four geometric methods
│   ├── evaluation.py       # metrics, confusion matrix, noise sweep
│   └── visualization.py    # matplotlib 3D (headless) + optional Open3D
├── notebooks/
│   └── mini_project.ipynb  # end-to-end, runnable
├── scripts/
│   └── run_experiment.py   # CLI: generate -> segment -> evaluate -> save figures
├── paper/
│   ├── main.tex            # IEEEtran conference paper starter
│   ├── references.bib
│   └── figures/            # (populated from results/ for the paper build)
├── results/                # generated figures + metrics.json
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone <your-repo-url>
cd staircase-pointcloud-segmentation
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

# Full experiment (writes figures + metrics.json to results/)
python scripts/run_experiment.py

# Or open the notebook
jupyter lab notebooks/mini_project.ipynb
```

Minimal programmatic use:

```python
from src.config import StaircaseConfig, SegmentationParams
from src.data_generation import generate_staircase
from src.segmentation import segment_by_normals
from src.evaluation import steppable_metrics

points, gt = generate_staircase(StaircaseConfig())
pred = segment_by_normals(points, SegmentationParams())
print(steppable_metrics(gt, pred))
```

## Reproducibility

All randomness is seeded (`StaircaseConfig.seed`, `SegmentationParams.rng_seed`).
`python scripts/run_experiment.py` reproduces every number in the table above.

## License

MIT — see [LICENSE](LICENSE).
