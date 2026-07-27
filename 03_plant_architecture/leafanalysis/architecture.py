"""Phase 2 — plant-architecture analytics for SAM3_new center-plant masks.

Pure ``mask -> traits`` (no torch, no SAM weights). Reads the existing
``<output_dir>/masks/*.mask.npy`` + ``*.meta.json`` produced by ``cli segment`` and the
aligned source RGB, then writes ``<output_dir>/architecture/architecture_report.{csv,json}``
plus a per-plant annotated figure.

Everything is computed only from the segmented center-plant mask (+ the photo pixels under it),
so neighbour/background plants are already excluded.

Scope (decided 2026-06-16, see ``PHASE2_ANALYTICS_PLAN.md``):
  * Tier 1 — robust size / shape / vertical-profile (bush habit) / colour-health traits.
  * Tier 2 — porosity, a 2D solid-of-revolution volume surrogate, texture, and colour
    flower/ripe-fruit *coverage* proxies (not counts — YOLO does counts downstream).
Lengths are reported in pixels; pass ``--scale CM_PER_PX`` to also emit cm columns. All
shape ratios and colour traits are scale-free and need no calibration.

Run:  ``python -m SAM3_new.architecture <output_dir> [--scale CM_PER_PX] [--no-figures]``
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.feature import graycomatrix, graycoprops

log = logging.getLogger("SAM3_new.architecture")

EPS = 1e-6

# Colour bins on OpenCV HSV (H:0-179, S:0-255, V:0-255). Heuristic and tunable; these are
# *coverage proxies*, not classifications. Fractions are independent and may overlap slightly.
_HSV_BINS = {
    "frac_green": lambda h, s, v: (h >= 36) & (h <= 89) & (s >= 40) & (v >= 40),
    "frac_yellow": lambda h, s, v: (h >= 20) & (h < 36) & (s >= 50) & (v >= 70),
    "frac_brown": lambda h, s, v: (h >= 5) & (h < 25) & (s >= 30) & (v >= 25) & (v < 120),
    "frac_red": lambda h, s, v: ((h <= 9) | (h >= 165)) & (s >= 60) & (v >= 40),
    "frac_blue_fruit": lambda h, s, v: (h >= 100) & (h <= 140) & (s >= 35) & (v >= 18),
    "frac_flower": lambda h, s, v: ((s < 35) & (v > 165)) | ((h >= 140) & (h < 165) & (s >= 30) & (v > 120)),
}


# --------------------------------------------------------------------------------------------
# Trait groups
# --------------------------------------------------------------------------------------------

def size_traits(region, mask: np.ndarray, base_xy) -> dict:
    """Tier 1 — extent in pixels. Canopy height = silhouette top-to-bottom (bbox vertical extent)."""
    minr, minc, maxr, maxc = region.bbox
    area = int(region.area)
    # Fraction of plant area below the stem-base anchor: a low-skirt / drooping-cane indicator.
    base_y = int(base_xy[1])
    skirt_below_base = round(float(mask[base_y:].sum()) / (area or 1), 4) if 0 <= base_y < mask.shape[0] else 0.0
    return {
        "area_px": area,
        "canopy_height_px": int(maxr - minr),
        "canopy_width_px": int(maxc - minc),
        "perimeter_px": round(float(region.perimeter), 2),
        "equiv_diameter_px": round(math.sqrt(4.0 * area / math.pi), 2),
        "feret_max_px": round(float(region.feret_diameter_max), 2),
        "skirt_below_base_frac": skirt_below_base,
    }


def shape_traits(region, mask: np.ndarray, cnt: np.ndarray) -> dict:
    """Tier 1 — scale-free shape descriptors (bush habit / compactness)."""
    area = float(region.area)
    minr, minc, maxr, maxc = region.bbox
    bbox_area = float((maxr - minr) * (maxc - minc)) or 1.0
    perim = cv2.arcLength(cnt, True) or EPS
    hull = cv2.convexHull(cnt)
    hull_perim = cv2.arcLength(hull, True) or EPS
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    rect_area = (rw * rh) or 1.0
    major = float(region.axis_major_length) or EPS
    minor = float(region.axis_minor_length)

    # log-scaled Hu moments (sign-preserving) for a compact rotation/scale-invariant signature
    hu = region.moments_hu
    hu_log = {f"hu{i + 1}": round(float(-np.sign(h) * np.log10(abs(h) + EPS)), 4) for i, h in enumerate(hu)}

    out = {
        "solidity": round(area / (float(region.area_convex) or 1.0), 4),
        "convex_hull_area_px": int(region.area_convex),
        "extent": round(area / bbox_area, 4),
        "aspect_ratio_hw": round((maxr - minr) / ((maxc - minc) or EPS), 4),
        "eccentricity": round(float(region.eccentricity), 4),
        "circularity": round(4.0 * math.pi * area / (perim * perim), 4),
        "convexity": round(min(hull_perim / perim, 1.0), 4),
        "rectangularity": round(area / rect_area, 4),
        "elongation": round(1.0 - (minor / major), 4),
        "major_axis_px": round(major, 2),
        "minor_axis_px": round(minor, 2),
        "orientation_deg": round(math.degrees(float(region.orientation)), 2),
    }
    out.update(hu_log)
    return out


def profile_traits(mask: np.ndarray, region, base_xy) -> dict:
    """Tier 1 — vertical/horizontal silhouette profiles: the core upright-vs-spreading signals."""
    w = mask.sum(axis=1).astype(np.float64)          # width per row
    rows = np.flatnonzero(w > 0)
    top_row, bot_row = int(rows[0]), int(rows[-1])
    wv = w[top_row:bot_row + 1]
    n = len(wv)
    band = max(1, n // 5)
    width_top = float(wv[:band].mean())
    width_base = float(wv[-band:].mean())
    width_mid = float(wv[2 * n // 5:3 * n // 5].mean()) if n >= 5 else float(wv.mean())
    argmax_from_top = int(np.argmax(wv))

    cy, cx = region.centroid
    base_x, base_y = float(base_xy[0]), float(base_xy[1])
    minr, minc, maxr, maxc = region.bbox
    bbox_w = (maxc - minc) or EPS
    plant_h = (bot_row - top_row) or EPS

    mid_row = (top_row + bot_row) // 2
    area_above = int(mask[:mid_row].sum())
    area_below = int(mask[mid_row:].sum())

    bx = int(round(base_x))
    bx = min(max(bx, 0), mask.shape[1] - 1)
    area_left = int(mask[:, :bx].sum())
    area_right = int(mask[:, bx:].sum())
    total = (area_left + area_right) or 1

    # 2D solid-of-revolution volume surrogate (px^3): revolve each row's width about a vertical axis.
    revolution_volume = float(np.sum(math.pi * (wv / 2.0) ** 2))

    return {
        "width_top_px": round(width_top, 1),
        "width_mid_px": round(width_mid, 1),
        "width_base_px": round(width_base, 1),
        "top_base_width_ratio": round(width_top / (width_base or EPS), 4),
        "widest_height_frac": round(1.0 - argmax_from_top / (n or 1), 4),  # 0=base, 1=top
        "lateral_lean": round((cx - base_x) / bbox_w, 4),                  # signed fraction of width
        "centroid_height_frac": round((bot_row - cy) / plant_h, 4),       # 0=bottom, 1=top
        "top_heaviness": round(area_above / (area_below or 1), 4),
        "lr_balance": round((area_right - area_left) / total, 4),         # signed; 0 = symmetric
        "asymmetry": round(abs(area_right - area_left) / total, 4),
        "revolution_volume_px3": round(revolution_volume, 1),
    }


def porosity_traits(mask: np.ndarray, region) -> dict:
    """Tier 2 — gaps / openness proxies (outline-level, not true interior light penetration)."""
    filled = ndimage.binary_fill_holes(mask)
    area = int(region.area)
    area_filled = int(filled.sum())
    holes = filled & ~mask
    lbl, _ = ndimage.label(holes)
    if lbl.max():
        sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, lbl.max() + 1))
        n_holes = int((sizes >= 25).sum())
    else:
        n_holes = 0
    return {
        "holes_area_px": area_filled - area,
        "holes_area_frac": round((area_filled - area) / (area_filled or 1), 4),
        "n_holes": n_holes,
        "gap_fraction": round(1.0 - area / (float(region.area_convex) or 1.0), 4),  # = 1 - solidity
    }


def color_traits(rgb_crop: np.ndarray, m: np.ndarray) -> dict:
    """Tier 1 colour-health indices + Tier 2 phenology coverage, over masked (eroded) pixels."""
    px = rgb_crop[m].astype(np.float64)
    if px.shape[0] == 0:
        return {}
    R, G, B = px[:, 0], px[:, 1], px[:, 2]
    s = R + G + B + EPS
    r, g, b = R / s, G / s, B / s

    exg = 2 * g - r - b
    gcc = G / s

    def mp(a):  # mean + 10/50/90 percentiles helper for headline indices
        p = np.percentile(a, [10, 50, 90])
        return round(float(a.mean()), 4), round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)

    exg_m, exg_p10, exg_p50, exg_p90 = mp(exg)
    gcc_m, gcc_p10, gcc_p50, gcc_p90 = mp(gcc)

    out = {
        "exg_mean": exg_m, "exg_p10": exg_p10, "exg_p50": exg_p50, "exg_p90": exg_p90,
        "exr_mean": round(float((1.4 * r - g).mean()), 4),
        "exgr_mean": round(float((exg - (1.4 * r - g)).mean()), 4),
        "gli_mean": round(float(((2 * G - R - B) / (2 * G + R + B + EPS)).mean()), 4),
        # VARI denominator can pass through zero -> compute on normalised channels and clip to [-1,1]
        "vari_mean": round(float(np.clip((g - r) / (g + r - b + EPS), -1.0, 1.0).mean()), 4),
        "ngrdi_mean": round(float(((G - R) / (G + R + EPS)).mean()), 4),
        "mgrvi_mean": round(float(((G ** 2 - R ** 2) / (G ** 2 + R ** 2 + EPS)).mean()), 4),
        "rgbvi_mean": round(float(((G ** 2 - B * R) / (G ** 2 + B * R + EPS)).mean()), 4),
        # TGI on 0-1 reflectance-scaled channels (keeps magnitude interpretable)
        "tgi_mean": round(float((-0.5 * (190 * (R - G) - 120 * (R - B)) / 255.0).mean()), 4),
        "gcc_mean": gcc_m, "gcc_p10": gcc_p10, "gcc_p50": gcc_p50, "gcc_p90": gcc_p90,
    }

    # HSV colour fractions + circular hue stats; Lab a*/b* means.
    hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2HSV)
    H = hsv[..., 0][m].astype(np.float64)
    S = hsv[..., 1][m].astype(np.float64)
    V = hsv[..., 2][m].astype(np.float64)
    npx = H.shape[0]
    for name, fn in _HSV_BINS.items():
        out[name] = round(float(fn(H, S, V).sum()) / npx, 4)

    ang = H / 180.0 * 2 * math.pi
    csum, ssum = np.cos(ang).mean(), np.sin(ang).mean()
    rlen = min(max(math.hypot(csum, ssum), EPS), 1.0)  # mean resultant length in (0, 1]
    out["hue_circ_mean_deg"] = round((math.degrees(math.atan2(ssum, csum)) % 360) / 2, 2)  # back to 0-179
    out["hue_circ_std"] = round(math.sqrt(-2 * math.log(rlen)), 4)

    lab = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2Lab)
    out["lab_a_mean"] = round(float(lab[..., 1][m].astype(np.float64).mean()) - 128, 2)
    out["lab_b_mean"] = round(float(lab[..., 2][m].astype(np.float64).mean()) - 128, 2)
    return out


def texture_traits(rgb_crop: np.ndarray, m: np.ndarray) -> dict:
    """Tier 2 — canopy texture on masked grayscale (relative; illumination-sensitive)."""
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    if not m.any():
        return {}
    gvals = gray[m].astype(np.float64)
    edges = cv2.Canny(gray, 50, 150)
    out = {
        "gray_mean": round(float(gvals.mean()), 2),
        "gray_std": round(float(gvals.std()), 2),
        "edge_density": round(float((edges[m] > 0).mean()), 4),
    }

    # GLCM on a downsampled, background-excluded, 16-level quantisation of the masked region.
    h, w = gray.shape
    f = max(1, int(math.ceil(max(h, w) / 256)))
    g_s = gray[::f, ::f]
    m_s = m[::f, ::f]
    if m_s.sum() < 16:
        return out
    levels = 16
    q = np.zeros(g_s.shape, dtype=np.uint8)
    gm = g_s[m_s].astype(np.float64)
    lo, hi = gm.min(), gm.max()
    scaled = np.zeros_like(g_s, dtype=np.float64)
    if hi > lo:
        scaled[m_s] = (g_s[m_s].astype(np.float64) - lo) / (hi - lo) * (levels - 1)
    q[m_s] = (scaled[m_s].astype(np.uint8) + 1)  # 1..levels; 0 reserved for background
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi / 2], levels=levels + 1, symmetric=True, normed=False)
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    if glcm.sum() == 0:
        return out
    out["glcm_contrast"] = round(float(graycoprops(glcm, "contrast").mean()), 4)
    out["glcm_homogeneity"] = round(float(graycoprops(glcm, "homogeneity").mean()), 4)
    out["glcm_energy"] = round(float(graycoprops(glcm, "energy").mean()), 4)
    out["glcm_correlation"] = round(float(graycoprops(glcm, "correlation").mean()), 4)
    p = glcm.astype(np.float64)
    p /= p.sum(axis=(0, 1), keepdims=True) + EPS
    out["glcm_entropy"] = round(float((-(p * np.log2(p + EPS)).sum(axis=(0, 1))).mean()), 4)
    return out


# --------------------------------------------------------------------------------------------
# Per-plant driver
# --------------------------------------------------------------------------------------------

def _largest_region(mask: np.ndarray):
    lbl = measure.label(mask, connectivity=2)
    props = measure.regionprops(lbl)
    if not props:
        return None
    return max(props, key=lambda r: r.area)


def analyze_plant(mask: np.ndarray, rgb: np.ndarray | None, meta: dict, scale_cm_per_px: float | None = None) -> dict:
    """Compute the full trait row for one plant. ``rgb`` may be None (geometry-only)."""
    stem = Path(meta["image_name"]).stem
    base_xy = tuple(meta.get("seed_xy", [mask.shape[1] // 2, mask.shape[0] - 1]))
    row: dict = {
        "stem": stem,
        "no_plant": bool(meta.get("no_plant", False)),
        "manually_edited": bool(meta.get("manually_edited", False)),
        "mask_confidence": round(float(meta.get("mask_confidence", 0.0)), 4),
        "img_height": int(meta.get("height", mask.shape[0])),
        "img_width": int(meta.get("width", mask.shape[1])),
    }
    if not mask.any():
        row["no_plant"] = True
        return row

    region = _largest_region(mask)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)

    row.update(size_traits(region, mask, base_xy))
    row.update(shape_traits(region, mask, cnt))
    row.update(profile_traits(mask, region, base_xy))
    row.update(porosity_traits(mask, region))

    if rgb is not None and rgb.shape[:2] == mask.shape:
        minr, minc, maxr, maxc = region.bbox
        rgb_crop = np.ascontiguousarray(rgb[minr:maxr, minc:maxc])
        m_crop = mask[minr:maxr, minc:maxc]
        # erode to keep colour stats off the soil/sky edge bleed
        k = max(1, int(round(0.01 * min(m_crop.shape))))
        m_eroded = cv2.erode(m_crop.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        if m_eroded.sum() < 50:
            m_eroded = m_crop
        row.update(color_traits(rgb_crop, m_eroded))
        row.update(texture_traits(rgb_crop, m_eroded))

    if scale_cm_per_px:
        sc = scale_cm_per_px
        row["canopy_height_cm"] = round(row["canopy_height_px"] * sc, 2)
        row["canopy_width_cm"] = round(row["canopy_width_px"] * sc, 2)
        row["area_cm2"] = round(row["area_px"] * sc * sc, 2)
    return row


# --------------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------------

def make_figure(stem: str, rgb: np.ndarray | None, mask: np.ndarray, region, base_xy, traits: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    H, W = mask.shape
    f = max(1, int(round(max(H, W) / 1100)))
    mask_s = mask[::f, ::f]
    minr, minc, maxr, maxc = region.bbox
    cy, cx = region.centroid

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), gridspec_kw={"width_ratios": [2, 1, 1]})

    ax = axes[0]
    if rgb is not None and rgb.shape[:2] == mask.shape:
        ax.imshow(rgb[::f, ::f])
    else:
        ax.imshow(mask_s, cmap="Greens")
    ax.contour(mask_s, levels=[0.5], colors="lime", linewidths=1.2)
    cnts, _ = cv2.findContours(mask_s.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea)).reshape(-1, 2)
        ax.plot(np.append(hull[:, 0], hull[0, 0]), np.append(hull[:, 1], hull[0, 1]), "y--", lw=1)
    ax.add_patch(Rectangle((minc / f, minr / f), (maxc - minc) / f, (maxr - minr) / f,
                           fill=False, edgecolor="cyan", lw=1.2))
    ax.plot(cx / f, cy / f, "r+", ms=14, mew=2, label="centroid")
    ax.plot(base_xy[0] / f, base_xy[1] / f, "m*", ms=14, label="stem base")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(stem)
    ax.axis("off")

    # width-vs-height profile
    w = mask.sum(axis=1).astype(float)
    rows = np.flatnonzero(w > 0)
    ax = axes[1]
    ax.plot(w[rows[0]:rows[-1] + 1], np.arange(rows[0], rows[-1] + 1), "g-")
    ax.invert_yaxis()
    ax.set_title("width profile w(y)")
    ax.set_xlabel("width (px)")
    ax.set_ylabel("row (y)")

    # colour fractions
    ax = axes[2]
    frac_keys = ["frac_green", "frac_yellow", "frac_brown", "frac_red", "frac_blue_fruit", "frac_flower"]
    colors = ["#2ca02c", "#ffd92f", "#8c564b", "#d62728", "#1f77b4", "#e377c2"]
    vals = [traits.get(k, 0.0) for k in frac_keys]
    ax.bar([k.replace("frac_", "") for k in frac_keys], vals, color=colors)
    ax.set_title("canopy colour fractions")
    ax.set_ylim(0, max(0.05, max(vals) * 1.2))
    ax.tick_params(axis="x", rotation=45)

    sub = (f"habit H/W={traits.get('aspect_ratio_hw','?')}  solidity={traits.get('solidity','?')}  "
           f"ExG={traits.get('exg_mean','?')}  green={traits.get('frac_green','?')}")
    fig.suptitle(sub, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def _load_rgb(meta: dict, mask_shape) -> np.ndarray | None:
    try:
        from .io_utils import load_upright
        rgb = load_upright(Path(meta["source_path"]), meta.get("rotate"))
        if rgb.shape[:2] != mask_shape:
            log.warning("RGB shape %s != mask %s for %s; skipping colour/texture",
                        rgb.shape[:2], mask_shape, meta.get("image_name"))
            return None
        return rgb
    except Exception as e:  # pragma: no cover - missing source image
        log.warning("Could not load RGB for %s: %s", meta.get("image_name"), e)
        return None


def run(output_dir: Path, scale_cm_per_px: float | None, figures: bool, limit: int | None,
        out_dir: Path | None = None) -> int:
    masks_dir = output_dir / "masks"
    if not masks_dir.is_dir():
        log.error("No masks/ directory under %s", output_dir)
        return 1
    arch_dir = out_dir if out_dir is not None else output_dir / "architecture"
    fig_dir = arch_dir / "figures"
    meta_paths = sorted(masks_dir.glob("*.meta.json"))
    if limit:
        meta_paths = meta_paths[:limit]
    if not meta_paths:
        log.error("No *.meta.json found under %s", masks_dir)
        return 1

    rows: list[dict] = []
    for i, mp in enumerate(meta_paths, 1):
        meta = json.loads(mp.read_text())
        stem = mp.name[: -len(".meta.json")]
        mask_path = masks_dir / f"{stem}.mask.npy"
        if not mask_path.exists():
            log.warning("[%d/%d] %s: mask missing", i, len(meta_paths), stem)
            continue
        mask = np.load(mask_path).astype(bool)
        rgb = _load_rgb(meta, mask.shape)
        log.info("[%d/%d] %s", i, len(meta_paths), stem)
        row = analyze_plant(mask, rgb, meta, scale_cm_per_px)
        rows.append(row)
        if figures and mask.any():
            region = _largest_region(mask)
            base_xy = tuple(meta.get("seed_xy", [mask.shape[1] // 2, mask.shape[0] - 1]))
            try:
                make_figure(stem, rgb, mask, region, base_xy, row, fig_dir / f"{stem}.png")
            except Exception as e:  # pragma: no cover
                log.warning("figure failed for %s: %s", stem, e)

    arch_dir.mkdir(parents=True, exist_ok=True)
    # union of keys, first-seen order, so optional colour/cm columns are preserved
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    csv_path = arch_dir / "architecture_report.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path = arch_dir / "architecture_report.json"
    json_path.write_text(json.dumps(rows, indent=2))

    print()
    print(f"Plants analysed:  {len(rows)}")
    print(f"Traits per plant: {len(fields)}")
    print(f"Report (CSV):     {csv_path}")
    print(f"Report (JSON):    {json_path}")
    if figures:
        print(f"Figures:          {fig_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="architecture.py",
                                description="Phase 2 plant-architecture analytics from existing masks.")
    p.add_argument("output_dir", help="The segmentation output dir (containing masks/), "
                                       "e.g. ../Segmentation/results")
    p.add_argument("--out-dir", default=None, dest="out_dir",
                   help="Where to write architecture_report.* + figures/ "
                        "(default: <output_dir>/architecture). Use e.g. ./results")
    p.add_argument("--scale", type=float, default=None, dest="scale_cm_per_px",
                   help="cm per pixel; if given, also emit cm/cm2 columns.")
    p.add_argument("--no-figures", action="store_true", help="Skip per-plant figures (faster).")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N plants.")
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    return run(Path(args.output_dir), args.scale_cm_per_px, not args.no_figures, args.limit,
               Path(args.out_dir) if args.out_dir else None)


if __name__ == "__main__":
    import sys
    sys.exit(main())
