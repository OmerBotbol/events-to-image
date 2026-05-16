#!/usr/bin/env python3
"""
results.py — Perceptual quality evaluation of a TWE event-camera reconstruction.

Pipeline
--------
1. Load GT (pure-black background) and reconstruction.
2. Detect tight bounding boxes of the zebra in each image.
3. Compute scale_factor = recon_bbox_height / gt_bbox_height.
4. Uniformly resize the entire GT by scale_factor (preserves aspect ratio).
5. HP-filter the scaled GT to match the reconstruction domain; normalize both.
6. Find X/Y translation with cv2.matchTemplate (no resize, ever).
7. Crop both to their exact overlapping region.
8. Compute LPIPS perceptual similarity and plot a 4-panel diagnostic figure.

Usage
-----
    python results.py --gt path/to/gt.jpg --recon events/exp_67
"""

import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

try:
    import torch
    import lpips as _lpips_mod  # type: ignore[import-untyped]
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# HP σ that matches tiwe.py's uniform_filter1d(size=1500) at subpixel_scale=100:
#   σ_box = size/√12  →  σ_px = (1500/√12)/100 ≈ 4.33 → round to 4.5
HP_SIGMA_DEFAULT       = 4.5
SUBPIXEL_SCALE_DEFAULT = 100   # tiwe.py: pix = round(x * 1e2)


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_gt(path: Path) -> np.ndarray:
    """Return GT as float32 grayscale in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read GT image: {path}")
    return img.astype(np.float32) / 255.0


def load_reconstruction(path: Path, subpixel_scale: int = SUBPIXEL_SCALE_DEFAULT) -> np.ndarray:
    """
    Load the TWE reconstruction as a raw float32 array.

    Prefers  reconstructed_raw.npy  inside the experiment folder (saved by
    tiwe.py as imageHP in sub-pixel space).  Columns are downsampled by
    `subpixel_scale` to convert back to camera-pixel resolution, and the
    array is negated so bright stripes → high values (matches the display
    convention: ax.imshow(-imageHP[:, ::scale])).

    Falls back to reading a PNG / JPEG directly.
    """
    folder   = path if path.is_dir() else path.parent
    npy_path = folder / "reconstructed_raw.npy"

    if npy_path.exists():
        arr = np.load(str(npy_path)).astype(np.float32)
        arr = -arr[:, ::subpixel_scale]
        print(f"  Loaded .npy   → {arr.shape[1]} × {arr.shape[0]} px "
              f"(÷{subpixel_scale} sub-pixel downsampling)")
        return arr

    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"  Loaded image  → {img.shape[1]} × {img.shape[0]} px")
            return img.astype(np.float32) / 255.0

    raise FileNotFoundError(
        f"No reconstructed_raw.npy found in {folder}, "
        f"and '{path}' is not a readable image file."
    )


# ─── Pre-processing helpers ───────────────────────────────────────────────────

def gaussian_highpass(img: np.ndarray, sigma: float) -> np.ndarray:
    """img − GaussianBlur(img, σ).  Matches tiwe.py's moving-average HP stage."""
    return img - gaussian_filter(img, sigma=sigma)


def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    return np.zeros_like(arr) if hi == lo else (arr - lo) / (hi - lo)


# ─── Bounding-box detection ───────────────────────────────────────────────────

def _find_bbox_gt(gt_float: np.ndarray) -> tuple[int, int, int, int]:
    """
    Locate the zebra in the GT image.

    The GT has a pure-black background (intensity ≈ 0), so a fixed low
    threshold cleanly separates foreground from background.  Morphological
    closing fills any internal gaps (gaps between stripes, shadows, etc.).

    Returns (x, y, w, h) of the tightest bounding rectangle.
    """
    u8 = (gt_float * 255).clip(0, 255).astype(np.uint8)
    _, binary = cv2.threshold(u8, 12, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = gt_float.shape
        return 0, 0, w, h
    return cv2.boundingRect(max(cnts, key=cv2.contourArea))


def _find_bbox_recon(recon: np.ndarray) -> tuple[int, int, int, int]:
    """
    Locate the zebra in the raw reconstruction.

    Background pixels have near-zero values (no events recorded there).
    Zebra pixels have large absolute deviations from the median.
    Otsu thresholding on the deviation map robustly separates the two.

    Returns (x, y, w, h) of the tightest bounding rectangle.
    """
    median    = float(np.median(recon))
    deviation = np.abs(recon - median).astype(np.float32)
    dev_u8    = (deviation / (deviation.max() + 1e-9) * 255).astype(np.uint8)

    _, binary = cv2.threshold(dev_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = recon.shape
        return 0, 0, w, h
    return cv2.boundingRect(max(cnts, key=cv2.contourArea))


# ─── Full alignment pipeline ──────────────────────────────────────────────────

def align_images(
    gt_raw: np.ndarray,
    recon: np.ndarray,
    hp_sigma: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Scale-match then translation-align GT and reconstruction.

    Returns
    -------
    gt_crop, recon_crop : float32 [0,1] arrays with identical shapes, ready
                          for LPIPS.
    info : dict with diagnostic values (bboxes, scale, match score, …).
    """

    # ── Step 1: bounding boxes ────────────────────────────────────────────────
    gx, gy, gw, gh = _find_bbox_gt(gt_raw)
    rx, ry, rw, rh = _find_bbox_recon(recon)
    print(f"  GT bbox    : x={gx}, y={gy}, w={gw}, h={gh}")
    print(f"  Recon bbox : x={rx}, y={ry}, w={rw}, h={rh}")

    # ── Step 2: scale factor ──────────────────────────────────────────────────
    scale  = rh / gh
    action = "downscaling" if scale < 1.0 else "upscaling"
    print(f"  Scale factor : {scale:.4f}  ({action} GT to match reconstruction)")

    # ── Step 3: uniform resize of the entire GT ───────────────────────────────
    new_h  = max(1, int(round(gt_raw.shape[0] * scale)))
    new_w  = max(1, int(round(gt_raw.shape[1] * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    gt_scaled = cv2.resize(gt_raw, (new_w, new_h), interpolation=interp)
    print(f"  GT after resize : {new_w} × {new_h} px")

    # ── Step 4: domain matching (HP filter GT) + normalize both ───────────────
    gt_norm  = normalize_01(gaussian_highpass(gt_scaled, sigma=hp_sigma))
    rec_norm = normalize_01(recon)

    # ── Step 5: matchTemplate for X/Y translation ─────────────────────────────
    #
    # matchTemplate requires the template to be ≤ the image in both dimensions.
    # We determine which image is smaller and use it as the template.
    h_g, w_g = gt_norm.shape
    h_r, w_r = rec_norm.shape

    if h_g <= h_r and w_g <= w_r:
        # GT (scaled down) fits inside reconstruction → search GT within recon
        result  = cv2.matchTemplate(rec_norm, gt_norm, cv2.TM_CCOEFF_NORMED)
        _, score, _, (tx, ty) = cv2.minMaxLoc(result)
        gt_crop    = gt_norm
        recon_crop = rec_norm[ty : ty + h_g, tx : tx + w_g]
        print(f"  GT is template → best match at ({tx}, {ty}) in recon frame")

    elif h_r <= h_g and w_r <= w_g:
        # Reconstruction fits inside scaled GT → search recon within GT
        result  = cv2.matchTemplate(gt_norm, rec_norm, cv2.TM_CCOEFF_NORMED)
        _, score, _, (tx, ty) = cv2.minMaxLoc(result)
        recon_crop = rec_norm
        gt_crop    = gt_norm[ty : ty + h_r, tx : tx + w_r]
        print(f"  Recon is template → best match at ({tx}, {ty}) in GT frame")

    else:
        # One axis fits but not the other — fall back to phaseCorrelate on a
        # shared canvas.  This handles partial-overlap cases.
        H = max(h_g, h_r);  W = max(w_g, w_r)

        def _pad(img: np.ndarray) -> np.ndarray:
            out = np.zeros((H, W), dtype=np.float32)
            out[: img.shape[0], : img.shape[1]] = img
            return out

        (fdx, fdy), _ = cv2.phaseCorrelate(_pad(gt_norm), _pad(rec_norm))
        ix, iy = int(round(fdx)), int(round(fdy))

        # Overlap bounds in reconstruction's frame
        rx0, rx1 = max(0,  ix), min(w_r,  ix + w_g)
        ry0, ry1 = max(0,  iy), min(h_r,  iy + h_g)
        # Equivalent bounds in GT's frame
        gx0, gx1 = rx0 - ix, rx1 - ix
        gy0, gy1 = ry0 - iy, ry1 - iy

        gt_crop    = gt_norm[gy0:gy1, gx0:gx1]
        recon_crop = rec_norm[ry0:ry1, rx0:rx1]
        tx, ty, score = ix, iy, 1.0
        print(f"  phaseCorrelate fallback → Δx={fdx:.1f}, Δy={fdy:.1f}")

    # ── Step 6: trim to identical shape (rounding safety) ─────────────────────
    h_ov = min(gt_crop.shape[0], recon_crop.shape[0])
    w_ov = min(gt_crop.shape[1], recon_crop.shape[1])
    gt_crop    = gt_crop[:h_ov, :w_ov]
    recon_crop = recon_crop[:h_ov, :w_ov]

    info = {
        "gt_bbox":       (gx, gy, gw, gh),
        "recon_bbox":    (rx, ry, rw, rh),
        "scale":         scale,
        "tx":            tx,
        "ty":            ty,
        "match_score":   float(score),
        "overlap_shape": (h_ov, w_ov),
    }
    return gt_crop, recon_crop, info


# ─── Pixel-level metrics ─────────────────────────────────────────────────────

def compute_pixel_metrics(ref: np.ndarray, cmp: np.ndarray) -> dict:
    """
    Compute MSE, PSNR, and SSIM on two float32 [0, 1] arrays of equal shape.

    Both arrays must already be perfectly aligned and cropped to the same region
    before calling this function.  data_range=1.0 is passed explicitly so that
    skimage does not try to infer it from the array dtype.
    """
    mse  = float(np.mean((ref - cmp) ** 2))
    psnr = float(peak_signal_noise_ratio(ref, cmp, data_range=1.0))
    ssim = float(structural_similarity(ref, cmp,   data_range=1.0))
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}


# ─── LPIPS ────────────────────────────────────────────────────────────────────

def _to_lpips_tensor(img_01: np.ndarray) -> "torch.Tensor":
    """float32 [0,1] grayscale → (1, 3, H, W) tensor in [−1, 1]."""
    t = torch.from_numpy(img_01).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    t = t.repeat(1, 3, 1, 1)                                  # (1,3,H,W)
    return t * 2.0 - 1.0


def compute_lpips(
    ref_01: np.ndarray, cmp_01: np.ndarray, net: str = "alex"
) -> tuple[float, np.ndarray]:
    """
    Returns (scalar_score, spatial_map_float32).
    spatial_map has the same H×W as the inputs; lower = more similar.
    """
    loss_fn = _lpips_mod.LPIPS(net=net, spatial=True, verbose=False)
    with torch.no_grad():
        spatial = loss_fn(_to_lpips_tensor(ref_01), _to_lpips_tensor(cmp_01))
    sp = spatial.squeeze().cpu().numpy()
    if sp.ndim == 3:
        sp = sp.mean(axis=0)
    if sp.shape != ref_01.shape:
        sp = cv2.resize(sp, (ref_01.shape[1], ref_01.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    return float(sp.mean()), sp.astype(np.float32)


# ─── Visualisation ────────────────────────────────────────────────────────────

def show_results(
    gt: np.ndarray, recon: np.ndarray,
    spatial_map: np.ndarray,
    lpips_score: float, pixel_metrics: dict, info: dict, net: str,
) -> plt.Figure:
    h_ov, w_ov = info["overlap_shape"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"LPIPS ({net}) = {lpips_score:.4f}   |   "
        f"PSNR = {pixel_metrics['PSNR']:.2f} dB   |   "
        f"SSIM = {pixel_metrics['SSIM']:.4f}   |   "
        f"scale = {info['scale']:.3f},  overlap = {w_ov} × {h_ov} px",
        fontsize=12,
    )

    axes[0].imshow(gt,    cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("GT  (HP-filtered, scaled, aligned)")
    axes[0].axis("off")

    axes[1].imshow(recon, cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("Reconstruction  (aligned)")
    axes[1].axis("off")

    axes[2].imshow(np.abs(gt - recon), cmap="hot", vmin=0, vmax=0.5, aspect="auto")
    axes[2].set_title("|Pixel diff|  (after alignment)")
    axes[2].axis("off")

    im = axes[3].imshow(spatial_map, cmap="plasma", vmin=0, aspect="auto")
    axes[3].set_title("LPIPS spatial map  (↓ = more similar)")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="LPIPS evaluation of TWE reconstruction vs GT, "
                    "with automatic scale-matching and translation alignment."
    )
    ap.add_argument("--gt",    required=True,
                    help="Path to the ground-truth image (JPG / PNG).")
    ap.add_argument("--recon", required=True,
                    help="Experiment folder (contains reconstructed_raw.npy) "
                         "or direct path to the reconstructed PNG.")
    ap.add_argument("--hp-sigma",       type=float, default=HP_SIGMA_DEFAULT,
                    help=f"Gaussian HP σ applied to the GT  (default: {HP_SIGMA_DEFAULT} px).")
    ap.add_argument("--subpixel-scale", type=int,   default=SUBPIXEL_SCALE_DEFAULT,
                    help=f"Sub-pixel scale used in tiwe.py  (default: {SUBPIXEL_SCALE_DEFAULT}).")
    ap.add_argument("--net",    default="alex", choices=["alex", "vgg", "squeeze"],
                    help="LPIPS backbone CNN  (default: alex).")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip the diagnostic figure.")
    args = ap.parse_args()

    if not LPIPS_AVAILABLE:
        print("WARNING: lpips / torch not found.  Run:  pip install lpips\n")

    # ── 1 / 3  Load ───────────────────────────────────────────────────────────
    print("\n[1/3]  Loading images …")
    gt_raw = load_gt(Path(args.gt))
    recon  = load_reconstruction(Path(args.recon), subpixel_scale=args.subpixel_scale)
    print(f"  GT:              {gt_raw.shape[1]} × {gt_raw.shape[0]} px")
    print(f"  Reconstruction:  {recon.shape[1]} × {recon.shape[0]} px")

    # ── 2 / 3  Scale-match + translate + crop ─────────────────────────────────
    print(f"\n[2/3]  Scale-matching and aligning  (σ_HP = {args.hp_sigma} px) …")
    gt_aligned, recon_aligned, info = align_images(gt_raw, recon, args.hp_sigma)

    h_ov, w_ov = info["overlap_shape"]
    score_flag = "reliable" if info["match_score"] > 0.3 else "LOW — inspect visually!"
    print(f"  Match score : {info['match_score']:.4f}  ({score_flag})")
    print(f"  Overlap     : {w_ov} × {h_ov} px")

    # Resolve the experiment folder for saving outputs
    recon_path = Path(args.recon)
    out_dir    = recon_path if recon_path.is_dir() else recon_path.parent

    # ── 3 / 4  Pixel-level metrics ────────────────────────────────────────────
    print("\n[3/4]  Computing pixel-level metrics (MSE / PSNR / SSIM) …")
    px = compute_pixel_metrics(gt_aligned, recon_aligned)
    print(f"  MSE  = {px['MSE']:.6f}")
    print(f"  PSNR = {px['PSNR']:.3f} dB")
    print(f"  SSIM = {px['SSIM']:.4f}")

    # ── 4 / 4  LPIPS ──────────────────────────────────────────────────────────
    print("\n[4/4]  Computing LPIPS …")
    if not LPIPS_AVAILABLE:
        print("  Skipped — install lpips first  (pip install lpips).")
        lpips_score, spatial_map = None, None
    else:
        lpips_score, spatial_map = compute_lpips(gt_aligned, recon_aligned, net=args.net)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("═" * 54)
    if lpips_score is not None:
        print(f"  LPIPS ({args.net:<6})  =  {lpips_score:.4f}   ← primary metric")
    print(f"  PSNR           =  {px['PSNR']:.3f} dB")
    print(f"  SSIM           =  {px['SSIM']:.4f}")
    print(f"  MSE            =  {px['MSE']:.6f}")
    print("═" * 54)
    print()
    print("  LPIPS interpretation:")
    print("    < 0.10   →  perceptually very similar  (excellent)")
    print("    0.10–0.25  →  good reconstruction")
    print("    0.25–0.45  →  noticeable differences")
    print("    > 0.45   →  poor perceptual match")
    print()

    # ── Save outputs ──────────────────────────────────────────────────────────
    metrics_record = {
        "LPIPS_net":  args.net,
        "LPIPS":      round(lpips_score, 6) if lpips_score is not None else None,
        "PSNR_dB":    round(px["PSNR"], 4),
        "SSIM":       round(px["SSIM"], 6),
        "MSE":        round(px["MSE"], 8),
        "scale":      round(info["scale"], 6),
        "match_score": round(info["match_score"], 4),
        "overlap_px": {"w": info["overlap_shape"][1], "h": info["overlap_shape"][0]},
        "hp_sigma":   args.hp_sigma,
    }
    metrics_path = out_dir / "quality_metrics.json"
    metrics_path.write_text(json.dumps(metrics_record, indent=2))
    print(f"  Metrics saved → {metrics_path}")

    if not args.no_plot and lpips_score is not None and spatial_map is not None:
        fig = show_results(gt_aligned, recon_aligned, spatial_map,
                           lpips_score, px, info, args.net)
        fig_path = out_dir / "quality_metrics.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved  → {fig_path}")
        plt.show()


if __name__ == "__main__":
    main()
