# Publishing this repository to GitHub

The project is already a local Git repository with an initial commit. You only
need to create an empty GitHub repo and push. Pick **one** of the two methods.

---

## Option A — GitHub CLI (`gh`), fastest

```bash
# 1. Install gh if needed:  https://cli.github.com
# 2. Authenticate once:
gh auth login

# 3. From the repo root, create the remote and push in one step:
cd staircase-pointcloud-segmentation
gh repo create staircase-pointcloud-segmentation \
    --public \
    --source=. \
    --remote=origin \
    --description "Geometric segmentation of steppable surfaces in simulated LiDAR staircase point clouds" \
    --push
```

That's it — the repo is live with full history.

---

## Option B — Manual (web + git), no CLI

1. Go to <https://github.com/new>.
2. Repository name: `staircase-pointcloud-segmentation`. Choose Public.
   **Do not** initialise with a README, .gitignore, or license (this repo
   already has them).
3. Click **Create repository**, then copy the HTTPS or SSH URL it shows.
4. From the repo root:

```bash
cd staircase-pointcloud-segmentation
git branch -M main
git remote add origin https://github.com/samaraqlan-G/staircase-pointcloud-segmentation.git
git push -u origin main
```

If you use SSH instead of HTTPS:

```bash
git remote add origin git@github.com:samaraqlan-G/staircase-pointcloud-segmentation.git
git push -u origin main
```

---

## If you downloaded the ZIP (no `.git` inside)

If your copy has no history, initialise it before pushing:

```bash
cd staircase-pointcloud-segmentation
git init
git add -A
git commit -m "Initial commit: staircase point-cloud steppable-surface segmentation"
git branch -M main
git remote add origin https://github.com/samaraqlan-G/staircase-pointcloud-segmentation.git
git push -u origin main
```

---

## After pushing

- **Add topics** (repo home → ⚙ next to About): `point-cloud`, `segmentation`,
  `ransac`, `dbscan`, `lidar`, `robotics`, `computer-vision`.
- **Set the About description** to the paper title.
- The figures in `results/` render inline in the README on GitHub, so the
  results table and images appear automatically.
- Optionally enable **GitHub Pages** (Settings → Pages → from `main`/`root`) if
  you want a simple project page.

## Recommended later commits

```bash
# Build and add the compiled paper PDF (kept out of git by .gitignore by default;
# force-add if you want it in the repo):
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
git add -f paper/main.pdf && git commit -m "Add compiled paper PDF"

# Add the presentation/video link to the README when ready.
```
