# TIWE — Temporal Image Warping Events

Python implementation of the TWE super-resolution algorithm for event-based cameras, based on:
> A. Stern, *"Suprima: Super-Resolution and Image Reconstruction in Event Cameras"*, BGU, 2025.

---

## Project Structure

```
event-to-image/
├── tiwe.py          # GUI application — main entry point
├── results.py       # Quality evaluation script (LPIPS, PSNR, SSIM, MSE)
├── main.py          # Headless script for a single experiment
├── convert.py       # Batch decoder for all experiments in events/
├── utils/
│   ├── decoder.py   # RAW → HDF5 conversion (standard and fast pipelines)
│   ├── flow.py      # Velocity estimation via contrast maximisation
│   ├── math.py      # Camera velocity and spatial-frequency helpers
│   └── cnvt_raw_2_csv/
│       └── build/Release/
│           └── metavision_evt3_raw_file_decoder.exe   # pre-compiled decoder
├── events/          # Put experiment folders here (ignored by git)
└── csv/             # Intermediate CSV files (ignored by git)
```

---

## Installation

### 1. Python dependencies

```bash
pip install numpy scipy h5py matplotlib opencv-python scikit-image lpips torch
```

> `lpips` and `torch` are only required for `results.py`. Everything else works without them.

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

### Step 2 — Reconstruct with the GUI (`tiwe.py`)

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

### Subpixel scale

Controls the resolution of the warped event grid. A scale of 100 means each camera pixel is divided into 100 sub-pixel columns. Higher values give finer spatial resolution at the cost of memory and compute time. The default of **100** is a good starting point; values between 50 and 200 are typical.

### HP filter window

The high-pass filter removes slow DC drift accumulated during the cumulative sum step. The window size is in subpixel columns. At scale 100, the default of **1500** corresponds to a Gaussian σ of ~4.3 camera pixels. Increase the window if the reconstruction shows large-scale brightness gradients.

### Normalisation divisor

Scales the cumulative sum image before high-pass filtering. The default of **65** is calibrated for typical event-camera contrast thresholds. If the reconstruction looks washed out, try a smaller value; if it is over-saturated, try a larger one.

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