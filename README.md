# Event-to-Image — Event Camera Reconstruction Toolkit

Python toolkit for reconstructing images from event-camera recordings.  Implements three algorithms:

| Script | Algorithm | Reference |
|--------|-----------|-----------|
| `tiwe.py` | TWE — Temporal Warping Events (super-resolution) | A. Stern, *Suprima*, BGU 2025 |
| `iwe.py` | IWE — Image of Warped Events (bilinear splatting) | Gallego et al., CVPR 2018 |
| `compare.py` | L2 / L1 / CNN inverse-problem reconstruction from IWE | Zhang et al., IEEE TPAMI 2022 |

---

## Project Structure

```
event-to-image/
├── tiwe.py            # TWE GUI — super-resolution reconstruction
├── iwe.py             # IWE GUI — warped-event accumulation
├── compare.py         # Comparison GUI — L2 / L1 / CNN reconstruction from IWE
├── results.py         # Quality evaluation script (LPIPS, PSNR, SSIM, MSE)
├── main.py            # Headless script for a single experiment
├── convert.py         # Batch decoder for all experiments in events/
├── models/            # Place drunet_gray.pth here for CNN reconstruction
├── utils/
│   ├── decoder.py     # RAW → HDF5 conversion (standard and fast pipelines)
│   ├── flow.py        # Velocity estimation via contrast maximisation (scan_vx)
│   ├── math.py        # Camera velocity and spatial-frequency helpers
│   ├── reconstruct.py # L2 / L1 / CNN solver functions
│   ├── drunet.py      # DRUNet (UNetRes) denoiser architecture
│   └── cnvt_raw_2_csv/
│       └── build/Release/
│           └── metavision_evt3_raw_file_decoder.exe   # pre-compiled decoder
├── events/            # Put experiment folders here (ignored by git)
└── csv/               # Intermediate CSV files (ignored by git)
```

---

## Installation

### 1. Python dependencies

```bash
pip install numpy scipy h5py matplotlib opencv-python scikit-image lpips torch pylops
```

| Package | Required by |
|---------|-------------|
| `numpy`, `scipy`, `h5py`, `matplotlib` | All scripts |
| `pylops` | `compare.py` (L2 / L1 solvers) |
| `torch` | `compare.py` (CNN solver) and `results.py` (LPIPS) |
| `lpips`, `opencv-python`, `scikit-image` | `results.py` only |

### 2. Metavision decoder

The standard decode path calls a pre-compiled C++ executable that converts EVT3 `.raw` files to CSV. It is shipped inside `utils/cnvt_raw_2_csv/build/Release/`. No additional installation is needed — just make sure that folder is present.

### 3. Experiment data

Copy each experiment folder (containing one or more `.raw` files) into the `events/` directory:

```
events/
└── exp_01/
    └── recording.raw
```

---

## Workflow

### Step 1 — Decode `.raw` to HDF5

Before any reconstruction you must convert the raw event-camera recording to HDF5 format.

**Option A — via the GUI (recommended)**

Run `tiwe.py` (see below), select the experiment folder, and click **Decode → .h5**.

**Option B — batch decode all experiments**

```bash
python convert.py
```

This iterates over every subfolder in `events/` and decodes any folder that does not already contain `events.h5`.

**Option C — programmatically**

```python
from pathlib import Path
from utils import decoder

decoder.runDecoder(Path("events/exp_01"))          # standard (RAW → CSV → HDF5)
decoder.runDecoderFast(Path("events/exp_01"))      # fast (RAW → HDF5 directly, needs expelliarmus)
```

The output file is always `events/<exp_folder>/events.h5`.

---

### Step 2a — Reconstruct with TWE (`tiwe.py`)

> TWE produces a **super-resolved intensity image** by integrating events along the motion direction.

```bash
python tiwe.py
```

The GUI walks you through the full pipeline:

| Control | Description |
|---------|-------------|
| **Browse** | Select the experiment folder (the one containing `events.h5`) |
| **Force re-decode** | Overwrite an existing `events.h5` when decoding |
| **Fast decode** | Skip the CSV intermediate — requires the `expelliarmus` package |
| **GPU arrays** | Use CuPy for GPU-accelerated decoding (requires CuPy) |
| **Decode → .h5** | Run the decoder for the selected folder |
| **Velocity vx (px/s)** | Horizontal velocity of the object in pixels per second |
| **Scan range / steps** | Search bounds and resolution for the velocity scan |
| **Scan vx** | Find the best vx automatically by maximising IWE contrast (recommended) |
| **Subpixel scale** | Integer multiplier applied to warped x coordinates (default: 100) |
| **HP filter window** | Moving-average window size for the high-pass filter in subpixel columns (default: 1500) |
| **Normalisation divisor** | Divisor applied after cumulative sum (default: 65) |
| **Run TWE** | Execute the full reconstruction and save the output |

#### Typical workflow in the GUI

1. Click **Browse** and select the experiment folder.
2. If `events.h5` does not yet exist, click **Decode → .h5**.
3. Click **Scan vx** — wait for the contrast-maximisation plot to appear, note the peak value (it is also written automatically into the vx field).
4. Adjust the vx field if needed, then click **Run TWE**.
5. The reconstructed image is displayed and saved to `events/<exp_folder>/reconstructed_image.png`. A raw float32 array is also saved as `reconstructed_raw.npy` for use by `results.py`.

#### Finding the right velocity

The **Scan vx** button sweeps a range of candidate velocities and selects the one that maximises the variance of the Image of Warped Events (IWE). A sharp, well-defined peak in the scan plot indicates a reliable estimate. If the scan range does not bracket the true velocity, adjust **min / max / steps** and re-scan.

---

### Step 2b — Reconstruct with IWE (`iwe.py`)

> IWE produces an **edge map** by warping events to a reference time and accumulating them via bilinear splatting.  The result can be fed into `compare.py` for full image reconstruction.

```bash
python iwe.py
```

| Control | Description |
|---------|-------------|
| **Browse** | Select the experiment folder (must contain `events.h5`) |
| **vx / Scan vx** | Horizontal velocity in px/s — use Scan vx to find it automatically |
| **Scan range** | Min / max / steps for the velocity scan |
| **vy** | Vertical velocity (0 for horizontal motion) |
| **Camera size W × H** | Sensor resolution — auto-detected if left at defaults |
| **Polarity mode** | *Signed* (±1) / *Count all* (unsigned) / *Split ±* (separate positive and negative channels) |
| **Compute IWE** | Run accumulation and save `iwe_reconstructed.npy` and `iwe_reconstructed.png` |

---

### Step 2c — Compare L2 / L1 / CNN reconstructions (`compare.py`)

> Solves the linear inverse problem `A·ℓ = b` where `b` is the IWE and `ℓ` is the recovered brightness image.  Three regularisation strategies are compared side by side.

```bash
python compare.py
```

**Prerequisites:**
- `events.h5` must exist in the folder (decoded in Step 1)
- `iwe_reconstructed.npy` must exist (computed in Step 2b)
- For CNN: download `drunet_gray.pth` (~64 MB) and place it in `models/`:
  ```
  https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth
  ```

#### Typical workflow

1. Browse to the experiment folder
2. Click **Scan vx** (or type the same `vx` used in `iwe.py`)
3. Click **▶ Run All & Compare**
4. Adjust parameters and re-run individual methods as needed

#### Output files

| File | Description |
|------|-------------|
| `reconstruction_comparison.png` | Side-by-side figure: IWE · L2 · L1 · CNN |
| `reconstructed_l2.npy` | L2 result as float32 array |
| `reconstructed_l1.npy` | L1 result as float32 array |
| `reconstructed_cnn.npy` | CNN result as float32 array (if model present) |

---

### Step 3 — Evaluate reconstruction quality (`results.py`)

Once a reconstruction exists, compare it against a ground-truth (GT) photograph:

```bash
python results.py --gt path/to/gt_photo.jpg --recon events/exp_01
```

#### What it does

1. Loads the GT image (expects a pure-black background around the subject).
2. Loads `reconstructed_raw.npy` from the experiment folder (or falls back to a PNG).
3. Detects the bounding box of the subject in each image and computes a scale factor.
4. Uniformly resizes the GT to match the reconstruction scale (no distortion).
5. Applies a Gaussian high-pass filter to the GT so both images are in the same edge-residual domain.
6. Uses `cv2.matchTemplate` to find the exact X/Y translation offset.
7. Crops both images to their strictly overlapping region (no resize at this step).
8. Computes **LPIPS**, **PSNR**, **SSIM**, and **MSE** on the aligned crops.
9. Saves the 4-panel diagnostic figure and a JSON metrics file to the experiment folder.

#### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--gt` | *(required)* | Path to the ground-truth image (JPG or PNG) |
| `--recon` | *(required)* | Experiment folder or direct path to the reconstructed PNG |
| `--hp-sigma` | `4.5` | Gaussian σ for high-pass filtering the GT (matches tiwe.py's default window) |
| `--subpixel-scale` | `100` | Sub-pixel scale used during reconstruction (must match tiwe.py setting) |
| `--net` | `alex` | LPIPS backbone: `alex`, `vgg`, or `squeeze` |
| `--no-plot` | off | Skip the interactive figure (still saves the PNG) |

#### Output files

| File | Description |
|------|-------------|
| `quality_metrics.json` | All scores (LPIPS, PSNR, SSIM, MSE), alignment diagnostics (scale, match score, overlap size) |
| `quality_metrics.png` | 4-panel figure: GT · Reconstruction · \|Pixel diff\| · LPIPS spatial map |

#### Interpreting the LPIPS score

| LPIPS | Interpretation |
|-------|---------------|
| < 0.10 | Perceptually very similar — excellent reconstruction |
| 0.10 – 0.25 | Good reconstruction |
| 0.25 – 0.45 | Noticeable differences |
| > 0.45 | Poor perceptual match |

---

### Headless single-experiment script (`main.py`)

`main.py` reproduces the core reconstruction pipeline without a GUI, useful for quick iteration or scripting:

```bash
python main.py
```

Edit the following constants at the top of the file before running:

| Variable | Description |
|----------|-------------|
| `events_path` | Path to the experiment folder |
| `vx` | Horizontal velocity in px/s (line 41) |
| ROI parameters in `calculate_object_velocity` | Physical camera setup (duration, resolution, ROI bounds) |

The script saves the reconstructed image to `events/<exp_folder>/reconstructed_image.png` and prints the CPP metric to the console.

---

## Parameter Reference

### Shared — optical flow

| Parameter | Default | Effect | How to choose |
|-----------|---------|--------|---------------|
| **vx (px/s)** | — | Horizontal velocity of the object across the sensor. The single most important parameter — wrong values give blurry or distorted output. | Use **Scan vx** first. Only enter manually if you know the physical speed. |
| **vy (px/s)** | `0.0` | Vertical velocity component. | Leave at 0 for horizontal motion. |
| **Scan range min/max** | `10` / `2000` | Bounds of the velocity search. | Widen the range if the scan plot has no clear peak. |
| **Scan steps** | `300` | Resolution of the search grid. | 100 is usually enough for a rough estimate; 300 for final use. |

---

### TWE (`tiwe.py`)

| Parameter | Default | Effect | How to choose |
|-----------|---------|--------|---------------|
| **Subpixel scale** | `100` | Each camera pixel is divided into this many sub-pixel columns in the warped image. Higher = finer spatial resolution but more memory and compute. | Start at 100. Increase to 200 if fine texture is lost; decrease to 50 if RAM is limited. |
| **HP filter window** | `1500` | Width of the moving-mean subtracted along the x-axis (in sub-pixel columns) to remove horizontal DC drift. | At scale 100, window 1500 ≈ 15 camera pixels. Increase if broad brightness gradients remain; decrease if real detail is being removed. |
| **Normalisation divisor** | `65` | The cumulative-sum image is divided by this value before high-pass filtering. Calibrated for typical event-camera contrast thresholds. | If the image looks washed out, decrease it. If over-saturated, increase it. |

---

### IWE (`iwe.py`)

| Parameter | Default | Effect | How to choose |
|-----------|---------|--------|---------------|
| **Camera size W × H** | `1280 × 720` | Output image dimensions. Auto-detected from the maximum event coordinates if left unchanged. | Leave at defaults unless your sensor is a different resolution. |
| **Polarity mode** | Signed | How event polarities are handled during accumulation. *Signed* (±1) highlights edges. *Count all* treats all events as +1 (event density). *Split ±* shows positive and negative channels separately. | *Signed* is best for feeding into `compare.py`. *Split* is useful for diagnosis. |

---

### L2 — Tikhonov (`compare.py`)

Solves `min‖A·ℓ − b‖² + λ‖∇²ℓ‖²` using LSQR.  Fast (seconds).  Produces soft, smooth edges.

| Parameter | Default | Effect | How to choose |
|-----------|---------|--------|---------------|
| **λ (reg. weight)** | `0.3` | Trade-off between fitting the IWE and smoothness. Larger → blurrier; smaller → noisier. | Start at 0.3. Halve or double until edges are sharp but not speckled. Typical range: `0.05`–`1.0`. |
| **LSQR iterations** | `100` | Solver steps. More = closer to the true solution. | 50 is often enough; increase to 200 if the result still looks under-converged. |
| **HP filter window** | `200` | Moving-mean window along x to remove horizontal banding (in camera pixels). 0 = disabled. | Set to roughly the width of the widest horizontal feature you want to *keep*. For a 1280 px image, 200–400 works well. |

---

### L1 — Total Variation (`compare.py`)

Solves `min‖A·ℓ − b‖² + λ(‖Dy·ℓ‖₁ + ‖Dx·ℓ‖₁)` using Split Bregman.  Slower (~minutes).  Promotes sharp edges and flat regions.

| Parameter | Default | Effect | How to choose |
|-----------|---------|--------|---------------|
| **λ (reg. weight)** | `0.1` | Trade-off between data fit and sparsity of gradients. Larger → flat "cartoon" look; smaller → noisy. L1 needs a smaller λ than L2 because TV is a stronger constraint. | Start at 0.1. Typical range: `0.01`–`0.3`. |
| **Outer iters** | `20` | Number of Split Bregman outer loops. More = better convergence but slower. | 10 for a quick look; 20–30 for final quality. |
| **Inner iters** | `5` | LSQR steps inside each outer loop. Keep this low. | 3–5 is sufficient; increasing beyond 10 rarely helps. |
| **HP filter window** | `200` | Same as L2. L1 usually produces less banding so you can use a larger window or disable it. | Same guidance as L2. |

---

### CNN — HQS + DRUNet (`compare.py`)

Alternates between gradient descent on the data term and a learned DRUNet denoiser.  Slowest (~minutes on GPU, hours on CPU).  Best perceptual quality.

| Parameter | Default | Effect | How to choose |
|-----------|---------|--------|---------------|
| **HQS iters** | `16` | How many times the algorithm alternates between gradient descent and denoising. More = better quality, slower. | Use 16 on GPU. On CPU, reduce to 6–8 for a practical runtime (~10–20 min). |
| **Grad iters** | `100` | Adam optimiser steps per HQS iteration. Controls how well the data term is minimised before each denoising step. | 80–120 on GPU; 30–50 on CPU. Diminishing returns above 150. |

> **CNN on CPU**: the code automatically falls back to CPU if no GPU is detected.  Reduce HQS iters to 6 and Grad iters to 30 for a run that completes in ~10–15 minutes.

---

### Quick tuning recipe

1. Run **Scan vx** first — everything else depends on a correct velocity estimate.
2. Run **L2** with defaults as a fast sanity check (seconds).
3. If the L2 result is blurry → decrease `λ`; if noisy → increase `λ`.
4. Adjust **HP filter window** until horizontal streaks disappear without softening the subject.
5. Run **L1** with a starting `λ` roughly 3× smaller than the L2 value that worked.
6. Run **CNN** last — it warm-starts from the L1 result, so good L1 parameters help CNN too.

---

## Troubleshooting

**`events.h5 not found` when clicking Run TWE**
Click **Decode → .h5** first, or run `convert.py` to batch-decode all experiments.

**Decode button does nothing / errors**
Ensure `utils/cnvt_raw_2_csv/build/Release/metavision_evt3_raw_file_decoder.exe` exists. If using Fast decode, install `expelliarmus` (`pip install expelliarmus`).

**Scan vx produces a flat plot with no clear peak**
The scan range may not bracket the true velocity. Widen the min/max range and increase the number of steps. Also check that the experiment folder contains a valid `events.h5`.

**LPIPS match score is LOW**
The subject bounding boxes may not be detected accurately. Check that the GT image has a genuinely pure-black background (no vignetting or noise) and that `reconstructed_raw.npy` exists rather than falling back to a PNG.

**`lpips` or `torch` import error in results.py**
```bash
pip install lpips torch
```
Without these, LPIPS is skipped but PSNR/SSIM/MSE still run.