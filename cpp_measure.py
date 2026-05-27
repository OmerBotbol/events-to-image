#!/usr/bin/env python3
"""
cpp_measure.py — Measure CPP (Cycles Per Pixel) of a bar-target group
                 in a TIWE event-camera reconstruction.

Given the full sub-pixel array saved by tiwe.py (reconstructed_raw.npy):
  1. Average a user-defined row band → noise-reduced 1D horizontal profile.
  2. Detect bar-target peaks with scipy.signal.find_peaks.
  3. Compute the first-to-last peak span in camera-pixel units.
  4. Calculate  CPP = N_CYCLES / span_camera_px.
  5. Plot the profile with annotated peaks and the CPP result.

Edit the "Configuration" block below, then run:
    python cpp_measure.py
    python cpp_measure.py --npy events/exp_42/reconstructed_raw.npy --scale 100 --cycles 4.5
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from pathlib import Path
from scipy.signal import find_peaks


# ══════════════════════════ Configuration ════════════════════════════════════
# Edit these values before running.  CLI flags override them at runtime.

# Path to the .npy file saved by tiwe.py
# NPY_PATH = Path("events/exp_70/reconstructed_raw.npy")
NPY_PATH = Path("C:\Electrical Engineering\Year 4\Engineering Project\Code\event-to-image\events\exp_70\reconstructed_raw.npy")

# Must match the "Subpixel scale" setting used in tiwe.py
SUBPIXEL_SCALE = 100

# ── ROI — specified in CAMERA pixels (converted to sub-pixel internally) ──────
# Rows: range of sensor rows to average (reduces noise; pick rows inside the bar group)
ROI_ROWS     = (200, 500)   # (row_start, row_end)

# Columns: rough bounds around the bar group of interest in CAMERA pixels
ROI_COLS_CAM = (300, 700)   # (col_start, col_end)

# ── Bar-target specification ──────────────────────────────────────────────────
N_CYCLES = 4.5   # number of cycles (black+white pairs) in the selected bar group

# ── Peak-detection tuning ─────────────────────────────────────────────────────
# Increase PROMINENCE if spurious noise peaks appear.
# Increase DISTANCE   if two neighbouring peaks of the same bar are merged.
PROMINENCE = 0.10   # fraction of the profile's full range (0–1)
DISTANCE   = 20     # minimum separation between peaks, in CAMERA pixels


ROI_ROWS     = (230, 270)   
ROI_COLS_CAM = (190, 215)   
PROMINENCE   = 0.10         
DISTANCE     = 20
# ═════════════════════════════════════════════════════════════════════════════


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_array(npy_path: Path) -> np.ndarray:
    """
    Load reconstructed_raw.npy and return the display-convention array.
    tiwe.py stores imageHP; the display shows −imageHP so that bright
    bars → high values.  We apply the same negation here.
    """
    return -np.load(str(npy_path)).astype(np.float32)


def extract_profile(
    image: np.ndarray,
    subpixel_scale: int,
    roi_rows: tuple[int, int],
    roi_cols_cam: tuple[int, int],
) -> tuple[np.ndarray, int, int]:
    """
    Crop the ROI and collapse it to a 1D horizontal profile by row-averaging.

    Parameters
    ----------
    image          : full 2D reconstruction  (rows × subpixel_cols)
    subpixel_scale : column scale factor
    roi_rows       : (row_start, row_end) — sensor rows to include
    roi_cols_cam   : (col_start, col_end) — camera pixels; converted internally

    Returns
    -------
    profile     : 1D float32 array, length = ROI width in subpixel cols
    col_start_sp: first subpixel column of the ROI (for absolute axis labelling)
    col_end_sp  : last  subpixel column of the ROI
    """
    r0, r1   = roi_rows
    c0_sp    = roi_cols_cam[0] * subpixel_scale
    c1_sp    = roi_cols_cam[1] * subpixel_scale

    # Clamp to array bounds
    r0    = max(0, r0);      r1    = min(image.shape[0], r1)
    c0_sp = max(0, c0_sp);   c1_sp = min(image.shape[1], c1_sp)

    roi     = image[r0:r1, c0_sp:c1_sp]
    profile = roi.mean(axis=0)
    return profile, c0_sp, c1_sp


# ─── Peak detection ───────────────────────────────────────────────────────────

def detect_peaks(
    profile: np.ndarray,
    prominence_frac: float,
    distance_cam_px: int,
    subpixel_scale: int,
) -> tuple[np.ndarray, float, int]:
    """
    Run scipy.signal.find_peaks on the profile.

    prominence_frac is expressed as a fraction of the profile's value range so
    the threshold is scale-invariant.  distance_cam_px is in camera pixels and
    is converted to subpixel columns internally.

    Returns (peak_indices, prominence_abs, distance_sp).
    """
    p_range      = float(profile.max() - profile.min())
    prominence_abs = prominence_frac * p_range
    distance_sp    = max(1, distance_cam_px * subpixel_scale)

    peaks, _ = find_peaks(profile, prominence=prominence_abs, distance=distance_sp)
    return peaks, prominence_abs, distance_sp


# ─── CPP calculation ──────────────────────────────────────────────────────────

def compute_cpp(
    peaks: np.ndarray,
    n_cycles: float,
    subpixel_scale: int,
) -> tuple[float, float, float]:
    """
    Calculate CPP from the first-to-last peak span.

    Returns (span_subpixels, span_camera_px, cpp).
    """
    span_sp     = float(peaks[-1] - peaks[0])
    span_cam    = span_sp / subpixel_scale
    cpp         = n_cycles / span_cam
    return span_sp, span_cam, cpp


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_results(
    profile: np.ndarray,
    peaks: np.ndarray,
    col_start_sp: int,
    subpixel_scale: int,
    n_cycles: float,
    span_sp: float | None,
    span_cam: float | None,
    cpp: float | None,
) -> None:
    # x-axis in camera pixels for human readability
    cam_axis = (np.arange(len(profile)) + col_start_sp) / subpixel_scale

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cam_axis, profile, lw=0.8, color="steelblue",
            label="Row-averaged profile")

    n_found = len(peaks)
    if n_found > 0:
        peak_cam = (peaks + col_start_sp) / subpixel_scale
        ax.plot(peak_cam, profile[peaks], "ro", ms=7, zorder=5,
                label=f"Detected peaks  ({n_found})")

        # Annotate each peak with its index
        for i, (px, py) in enumerate(zip(peak_cam, profile[peaks])):
            ax.annotate(str(i + 1), xy=(px, py),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7, color="darkred")

        if n_found >= 2:
            ax.axvspan(peak_cam[0], peak_cam[-1], alpha=0.08, color="tomato",
                       label=f"Measured span  ({span_cam:.2f} px)")

    # Info box
    if cpp is not None:
        info_lines = [
            f"Detected peaks :  {n_found}",
            f"Span (subpixel):  {span_sp:.0f}  sp-cols",
            f"Span (camera)  :  {span_cam:.4f}  px",
            f"N cycles       :  {n_cycles}",
            f"CPP            :  {cpp:.5f}  cycles / px",
            f"Nyquist limit  :  0.5000  cycles / px",
        ]
        ax.text(
            0.98, 0.97, "\n".join(info_lines),
            transform=ax.transAxes, fontsize=9,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="lightyellow", edgecolor="gray", alpha=0.93),
        )
        title = (f"Bar-target CPP analysis  —  "
                 f"CPP = {cpp:.5f} cycles / camera-px   "
                 f"(Nyquist = 0.5)")
    else:
        title = "Bar-target CPP analysis  —  not enough peaks detected (need ≥ 2)"

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Camera-pixel position")
    ax.set_ylabel("Mean intensity  (a.u.)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()


# ─── GUI ──────────────────────────────────────────────────────────────────────

class CPPApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("CPP Measure — Bar Target Analysis")
        self.resizable(False, False)
        self._build_ui()

    def _build_ui(self):
        P = {"padx": 8, "pady": 4}

        # ── File row ──────────────────────────────────────────────────────────
        tk.Label(self, text="Reconstruction file (.npy):").grid(
            row=0, column=0, sticky="w", **P)
        self.npy_var = tk.StringVar(value=str(NPY_PATH))
        tk.Entry(self, textvariable=self.npy_var, width=52).grid(
            row=0, column=1, columnspan=2, **P)
        tk.Button(self, text="Browse…", command=self._browse).grid(
            row=0, column=3, **P)

        ttk.Separator(self, orient="horizontal").grid(
            row=1, column=0, columnspan=4, sticky="ew", pady=6)

        # ── Algorithm parameters ──────────────────────────────────────────────
        tk.Label(self, text="Subpixel scale:").grid(
            row=2, column=0, sticky="w", **P)
        self.scale_var = tk.StringVar(value=str(SUBPIXEL_SCALE))
        tk.Entry(self, textvariable=self.scale_var, width=10).grid(
            row=2, column=1, sticky="w", **P)
        tk.Label(self, text="Must match the value used in tiwe.py",
                 foreground="gray").grid(row=2, column=2, sticky="w", **P)

        tk.Label(self, text="Number of cycles:").grid(
            row=3, column=0, sticky="w", **P)
        self.cycles_var = tk.StringVar(value=str(N_CYCLES))
        tk.Entry(self, textvariable=self.cycles_var, width=10).grid(
            row=3, column=1, sticky="w", **P)
        tk.Label(self, text="Cycles in the selected bar group",
                 foreground="gray").grid(row=3, column=2, sticky="w", **P)

        ttk.Separator(self, orient="horizontal").grid(
            row=4, column=0, columnspan=4, sticky="ew", pady=6)

        # ── ROI ───────────────────────────────────────────────────────────────
        tk.Label(self, text="ROI rows:").grid(
            row=5, column=0, sticky="w", **P)
        roi_row_frame = tk.Frame(self)
        roi_row_frame.grid(row=5, column=1, columnspan=2, sticky="w")
        tk.Label(roi_row_frame, text="from").pack(side="left")
        self.row0_var = tk.StringVar(value=str(ROI_ROWS[0]))
        tk.Entry(roi_row_frame, textvariable=self.row0_var, width=7).pack(
            side="left", padx=4)
        tk.Label(roi_row_frame, text="to").pack(side="left")
        self.row1_var = tk.StringVar(value=str(ROI_ROWS[1]))
        tk.Entry(roi_row_frame, textvariable=self.row1_var, width=7).pack(
            side="left", padx=4)
        tk.Label(roi_row_frame, text="(sensor rows)",
                 foreground="gray").pack(side="left", padx=4)

        tk.Label(self, text="ROI columns:").grid(
            row=6, column=0, sticky="w", **P)
        roi_col_frame = tk.Frame(self)
        roi_col_frame.grid(row=6, column=1, columnspan=2, sticky="w")
        tk.Label(roi_col_frame, text="from").pack(side="left")
        self.col0_var = tk.StringVar(value=str(ROI_COLS_CAM[0]))
        tk.Entry(roi_col_frame, textvariable=self.col0_var, width=7).pack(
            side="left", padx=4)
        tk.Label(roi_col_frame, text="to").pack(side="left")
        self.col1_var = tk.StringVar(value=str(ROI_COLS_CAM[1]))
        tk.Entry(roi_col_frame, textvariable=self.col1_var, width=7).pack(
            side="left", padx=4)
        tk.Label(roi_col_frame, text="(camera pixels)",
                 foreground="gray").pack(side="left", padx=4)

        ttk.Separator(self, orient="horizontal").grid(
            row=7, column=0, columnspan=4, sticky="ew", pady=6)

        # ── Peak detection ────────────────────────────────────────────────────
        tk.Label(self, text="Peak prominence:").grid(
            row=8, column=0, sticky="w", **P)
        self.prom_var = tk.StringVar(value=str(PROMINENCE))
        tk.Entry(self, textvariable=self.prom_var, width=10).grid(
            row=8, column=1, sticky="w", **P)
        tk.Label(self, text="Fraction of profile range (0–1).  Raise to suppress noise peaks.",
                 foreground="gray").grid(row=8, column=2, sticky="w", **P)

        tk.Label(self, text="Peak distance:").grid(
            row=9, column=0, sticky="w", **P)
        self.dist_var = tk.StringVar(value=str(DISTANCE))
        tk.Entry(self, textvariable=self.dist_var, width=10).grid(
            row=9, column=1, sticky="w", **P)
        tk.Label(self, text="Minimum spacing between peaks in camera pixels.",
                 foreground="gray").grid(row=9, column=2, sticky="w", **P)

        ttk.Separator(self, orient="horizontal").grid(
            row=10, column=0, columnspan=4, sticky="ew", pady=6)

        # ── Status + result ───────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(self, textvariable=self.status_var,
                 foreground="blue", wraplength=520, anchor="w").grid(
            row=11, column=0, columnspan=4, sticky="w", **P)

        self.result_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.result_var,
                 foreground="darkgreen", font=("Courier", 10, "bold"),
                 anchor="w").grid(
            row=12, column=0, columnspan=4, sticky="w", **P)

        # ── Run button ────────────────────────────────────────────────────────
        tk.Button(
            self, text="▶  Run Analysis", command=self._run,
            bg="#4CAF50", fg="white", font=("Arial", 11, "bold"),
            padx=20, pady=8,
        ).grid(row=13, column=0, columnspan=4, pady=14)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select reconstructed_raw.npy",
            filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
        )
        if path:
            self.npy_var.set(path)

    def _status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def _run(self):
        # ── Parse inputs ──────────────────────────────────────────────────────
        try:
            npy_path   = Path(self.npy_var.get().strip())
            scale      = int(self.scale_var.get())
            n_cycles   = float(self.cycles_var.get())
            row0       = int(self.row0_var.get())
            row1       = int(self.row1_var.get())
            col0       = int(self.col0_var.get())
            col1       = int(self.col1_var.get())
            prominence = float(self.prom_var.get())
            distance   = int(self.dist_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        if not npy_path.exists():
            messagebox.showerror("File not found", f"{npy_path}")
            return

        # ── Load ──────────────────────────────────────────────────────────────
        self._status(f"Loading  {npy_path} …")
        self.result_var.set("")
        try:
            image = load_array(npy_path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            self._status("Load failed.")
            return

        total_rows, total_sp = image.shape
        self._status(
            f"Loaded  {total_sp} subpixel cols × {total_rows} rows  "
            f"({total_sp // scale} × {total_rows} camera px).  "
            f"Extracting profile…"
        )

        # ── Extract profile ───────────────────────────────────────────────────
        profile, c0_sp, c1_sp = extract_profile(
            image, scale, (row0, row1), (col0, col1)
        )
        print(f"\nROI  rows {row0}–{row1},  cols {col0}–{col1} camera px  "
              f"→  subpixel {c0_sp}–{c1_sp}")
        print(f"Profile length : {len(profile)} sp-cols  "
              f"= {len(profile) / scale:.1f} camera px")

        # ── Detect peaks ──────────────────────────────────────────────────────
        self._status("Detecting peaks…")
        peaks, prom_abs, dist_sp = detect_peaks(profile, prominence, distance, scale)
        print(f"Prominence threshold : {prom_abs:.5f}  ({prominence:.0%} of range)")
        print(f"Min distance         : {dist_sp} sp-cols  = {distance} camera px")
        print(f"Peaks found          : {len(peaks)}")

        if len(peaks) < 2:
            self._status(
                f"Only {len(peaks)} peak(s) detected — need ≥ 2.  "
                "Widen ROI columns, or reduce Prominence / Distance."
            )
            self.result_var.set("")
            plot_results(profile, peaks, c0_sp, scale, n_cycles, None, None, None)
            return

        # ── Compute CPP ───────────────────────────────────────────────────────
        span_sp, span_cam, cpp = compute_cpp(peaks, n_cycles, scale)

        print(f"\nResults:")
        print(f"  First peak : sp-col {peaks[0]}  = camera px {(peaks[0]+c0_sp)/scale:.3f}")
        print(f"  Last  peak : sp-col {peaks[-1]}  = camera px {(peaks[-1]+c0_sp)/scale:.3f}")
        print(f"  Span       : {span_sp:.0f} sp-cols  = {span_cam:.4f} camera px")
        print(f"  CPP        : {cpp:.5f} cycles / camera-pixel")

        self._status(
            f"Done.   {len(peaks)} peaks detected   |   "
            f"Span = {span_cam:.3f} px   |   CPP = {cpp:.5f} cycles/px"
        )
        self.result_var.set(
            f"CPP = {cpp:.5f} cycles / camera-pixel"
            f"    (span = {span_cam:.3f} px,  {len(peaks)} peaks)"
        )

        plot_results(profile, peaks, c0_sp, scale, n_cycles, span_sp, span_cam, cpp)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = CPPApp()
    app.mainloop()
