"""
TIWE — Temporal Image Warping Events
Implements the TWE super-resolution algorithm from:
  A. Stern, "Suprima: Super-Resolution and Image Reconstruction in Event Cameras", BGU, 2025.

Usage:  python tiwe.py
"""
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.sparse import coo_matrix
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils import decoder


# ── helpers ──────────────────────────────────────────────────────────────────

def load_events(hdf5_path: Path):
    with h5py.File(hdf5_path, "r") as f:
        evs = f["events"]
        t   = evs["timestamp"][:].astype(np.float64)
        x   = evs["x"][:].astype(np.float32)
        y   = evs["y"][:].astype(np.int32)
        pol = evs["polarity"][:].astype(np.float32)
    return t, x, y, pol


def run_twe(t, x, y, polarity, vx: float, subpixel_scale: int,
            hp_window: int, norm_div: float):
    """
    Apply Temporal Event Warping and reconstruct a log-intensity image.

    Steps follow the paper:
      1. Warp timestamps to a common spatial reference  (eq. 17)
      2. Map warped x to an integer subpixel grid
      3. Accumulate polarity into a 2-D sparse histogram  (eq. 16)
      4. Cumulative sum along the x-axis  →  log-intensity trace  (eq. 9)
      5. Normalise
      6. High-pass filter to remove DC drift / threshold imbalance  (sec. III-H)

    Returns
    -------
    imageHP : np.ndarray  shape (rows, subpixel_cols)
    """
    # --- (1) TWE: x̂_k = x_k + v·(t_ref − t_k)  (eq. 17) ---
    t_ref = float(t[-1])
    dt    = (t - t_ref).astype(np.float32)   # t_k − t_ref  (negative for past events)
    x    -= dt * np.float32(vx)              # x̂ = x + v·(t_ref − t_k)
    del dt, t

    # --- (2) Map to integer subpixel grid ---
    pix = np.round(x * np.float32(subpixel_scale)).astype(np.int32)
    del x
    pix -= pix.min()

    # --- (3) Accumulate polarity into 2-D image ---
    num_rows = int(y.max()) + 1
    num_cols = int(pix.max()) + 1
    image = coo_matrix(
        (polarity, (y, pix)), shape=(num_rows, num_cols)
    ).toarray()
    del polarity, pix, y

    # --- (4) Integrate along the warped-x axis  (cumulative polarity sum) ---
    image = np.cumsum(image, axis=1)

    # --- (5) Normalise ---
    image /= norm_div

    # --- (6) High-pass filter (subtract slow drift) ---
    moving_avg = uniform_filter1d(image, size=hp_window, axis=1, mode="nearest")
    imageHP    = image - moving_avg
    del image, moving_avg

    return imageHP


# ── GUI ───────────────────────────────────────────────────────────────────────

class TWEApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("TIWE — Temporal Image Warping Events")
        self.resizable(False, False)
        self._build_ui()

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        P = {"padx": 8, "pady": 4}

        # ── folder row ──
        tk.Label(self, text="Events folder (.raw files):").grid(
            row=0, column=0, sticky="w", **P)
        self.folder_var = tk.StringVar()
        tk.Entry(self, textvariable=self.folder_var, width=52).grid(
            row=0, column=1, **P)
        tk.Button(self, text="Browse…", command=self._browse).grid(
            row=0, column=2, **P)

        # ── decode options + Decode button ──
        decode_frame = tk.Frame(self)
        decode_frame.grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=2)

        self.force_decode_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            decode_frame, text="Force re-decode  (overwrite existing events.h5)",
            variable=self.force_decode_var,
        ).pack(side="left")

        self.fast_decode_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            decode_frame, text="Fast decode  (no CSV — needs expelliarmus)",
            variable=self.fast_decode_var,
        ).pack(side="left", padx=12)

        self.gpu_decode_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            decode_frame, text="GPU arrays  (needs CuPy)",
            variable=self.gpu_decode_var,
        ).pack(side="left")

        tk.Button(
            decode_frame, text="Decode  →  .h5", command=self._decode,
            bg="#FF9800", fg="white", font=("Arial", 9, "bold"),
        ).pack(side="left", padx=16)

        ttk.Separator(self, orient="horizontal").grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=6)

        # ── velocity ──
        tk.Label(self, text="Velocity  vx  (px/s):").grid(
            row=3, column=0, sticky="w", **P)
        self.vx_var = tk.StringVar(value="")
        tk.Entry(self, textvariable=self.vx_var, width=15).grid(
            row=3, column=1, sticky="w", **P)
        tk.Label(
            self,
            text="Enter manually or use 'Scan vx' below  (required)",
            foreground="gray",
        ).grid(row=4, column=1, sticky="w", padx=8)

        # ── velocity scan ──
        scan_frame = tk.Frame(self)
        scan_frame.grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=2)
        tk.Label(scan_frame, text="Scan range:").pack(side="left")
        tk.Label(scan_frame, text=" min").pack(side="left")
        self.scan_min_var = tk.StringVar(value="10")
        tk.Entry(scan_frame, textvariable=self.scan_min_var, width=8).pack(side="left")
        tk.Label(scan_frame, text=" max").pack(side="left")
        self.scan_max_var = tk.StringVar(value="2000")
        tk.Entry(scan_frame, textvariable=self.scan_max_var, width=8).pack(side="left")
        tk.Label(scan_frame, text=" steps").pack(side="left")
        self.scan_steps_var = tk.StringVar(value="300")
        tk.Entry(scan_frame, textvariable=self.scan_steps_var, width=6).pack(side="left")
        tk.Button(
            scan_frame, text="Scan vx", command=self._scan_vx,
            bg="#2196F3", fg="white", font=("Arial", 9, "bold"),
        ).pack(side="left", padx=6)

        # ── subpixel scale ──
        tk.Label(self, text="Subpixel scale:").grid(
            row=6, column=0, sticky="w", **P)
        self.scale_var = tk.StringVar(value="100")
        tk.Entry(self, textvariable=self.scale_var, width=10).grid(
            row=6, column=1, sticky="w", **P)
        tk.Label(
            self,
            text="Integer multiplier on warped x  (higher → finer grid)",
            foreground="gray",
        ).grid(row=7, column=1, sticky="w", padx=8)

        # ── HP filter window ──
        tk.Label(self, text="HP filter window (cols):").grid(
            row=8, column=0, sticky="w", **P)
        self.hp_var = tk.StringVar(value="1500")
        tk.Entry(self, textvariable=self.hp_var, width=10).grid(
            row=8, column=1, sticky="w", **P)

        # ── normalisation ──
        tk.Label(self, text="Normalisation divisor:").grid(
            row=9, column=0, sticky="w", **P)
        self.norm_var = tk.StringVar(value="65")
        tk.Entry(self, textvariable=self.norm_var, width=10).grid(
            row=9, column=1, sticky="w", **P)

        ttk.Separator(self, orient="horizontal").grid(
            row=10, column=0, columnspan=3, sticky="ew", pady=6)

        # ── status + progress ──
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(
            self, textvariable=self.status_var,
            foreground="blue", wraplength=460, anchor="w",
        ).grid(row=11, column=0, columnspan=3, sticky="w", **P)

        self.progress = ttk.Progressbar(self, mode="indeterminate", length=460)
        self.progress.grid(row=12, column=0, columnspan=3, padx=8, pady=4)

        # ── run button ──
        tk.Button(
            self, text="▶  Run TWE", command=self._run,
            bg="#4CAF50", fg="white", font=("Arial", 11, "bold"),
            padx=20, pady=8,
        ).grid(row=13, column=0, columnspan=3, pady=14)

    # ── callbacks ────────────────────────────────────────────────────────────

    def _browse(self):
        folder = filedialog.askdirectory(title="Select folder containing .raw files")
        if folder:
            self.folder_var.set(folder)

    def _status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def _hdf5_path(self) -> Path:
        return Path(self.folder_var.get().strip()) / "events.h5"

    def _decode(self):
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showerror("Error", "Select a folder first.")
            return
        events_path = Path(folder)
        self.progress.start()
        try:
            if self.fast_decode_var.get():
                self._status("Fast-decoding .raw → events.h5  (no CSV)…")
                decoder.runDecoderFast(events_path, use_gpu=self.gpu_decode_var.get())
            else:
                self._status("Decoding .raw → events.h5…")
                decoder.runDecoder(events_path)

            h5_path = events_path / "events.h5"
            if h5_path.exists():
                self._status(f"Decode complete  →  {h5_path}")
            else:
                messagebox.showerror(
                    "Decode failed",
                    "events.h5 was not created. Check that the folder contains .raw "
                    "files and that the decoder executable is present.",
                )
                self._status("Decode failed.")
        except Exception as exc:
            messagebox.showerror("Decode failed", str(exc))
            self._status(f"Decode error: {exc}")
        finally:
            self.progress.stop()

    def _scan_vx(self):
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showerror("Error", "Select a folder first.")
            return
        h5 = Path(folder) / "events.h5"
        if not h5.exists():
            messagebox.showerror("Error", "events.h5 not found. Run decoding first.")
            return
        try:
            vx_min  = float(self.scan_min_var.get())
            vx_max  = float(self.scan_max_var.get())
            n_steps = int(self.scan_steps_var.get())
            if vx_min >= vx_max or n_steps < 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid scan range", "Check min < max and steps ≥ 2.")
            return

        self._status("Loading events for velocity scan…")
        self.progress.start()
        try:
            t, x, y, _ = load_events(h5)

            # Window: object travels ~300 px at the *fastest* candidate speed.
            # This keeps the window short regardless of what vx_min is.
            total_duration = float(t[-1] - t[0])
            window_s = min(300.0 / max(vx_max, 1.0), total_duration)
            t_mid = (float(t[0]) + float(t[-1])) / 2.0
            mask  = (t >= t_mid - window_s / 2) & (t <= t_mid + window_s / 2)

            if mask.sum() == 0:
                # Window too narrow or events not centred — use the full recording
                t_w = t.astype(np.float64)
                x_w = x.astype(np.float64)
                y_w = y.astype(np.float64)
                window_s = total_duration
            else:
                t_w = t[mask].astype(np.float64)
                x_w = x[mask].astype(np.float64)
                y_w = y[mask].astype(np.float64)
            del t, x, y  # free full-recording arrays immediately

            # Subsample to at most MAX_EVENTS so each histogram step is fast.
            MAX_EVENTS = 100_000
            if len(t_w) > MAX_EVENTS:
                rng = np.random.default_rng(0)
                idx = np.sort(rng.choice(len(t_w), MAX_EVENTS, replace=False))
                t_w, x_w, y_w = t_w[idx], x_w[idx], y_w[idx]

            n_events_used = len(t_w)
            t_ref  = float(t_w[-1])
            x_lo, x_hi = float(x_w.min()), float(x_w.max())
            y_lo, y_hi = float(y_w.min()), float(y_w.max())
            x_bins = max(int(x_hi - x_lo), 1)
            y_bins = max(int(y_hi - y_lo), 1)

            candidates = np.linspace(vx_min, vx_max, n_steps)
            variances  = np.empty(n_steps)

            for i, vx_c in enumerate(candidates):
                x_warped = x_w - (t_w - t_ref) * vx_c
                iwe, _, _ = np.histogram2d(
                    x_warped, y_w,
                    bins=[x_bins, y_bins],
                    range=[[x_lo, x_hi], [y_lo, y_hi]],
                )
                variances[i] = np.var(iwe)
                if i % 20 == 0:
                    self._status(f"Scanning… {i}/{n_steps}  ({100*i//n_steps}%)")

            best_vx = float(candidates[np.argmax(variances)])
            self.vx_var.set(f"{best_vx:.4f}")
            self._status(
                f"Scan complete.  Best vx = {best_vx:.4f} px/s  "
                f"(window = {window_s*1000:.0f} ms,  {n_events_used:,} events used)"
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(candidates, variances, linewidth=1)
            ax.axvline(best_vx, color="r", linestyle="--",
                       label=f"Peak  {best_vx:.2f} px/s")
            ax.set_xlabel("Candidate  vx  (px/s)")
            ax.set_ylabel("IWE variance  (contrast)")
            ax.set_title("Velocity scan — contrast maximisation landscape")
            ax.legend()
            fig.tight_layout()
            plt.show()

        except Exception as exc:
            messagebox.showerror("Scan failed", str(exc))
            self._status(f"Scan error: {exc}")
        finally:
            self.progress.stop()

    def _run(self):
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showerror("Error", "Select a folder first.")
            return
        events_path = Path(folder)

        try:
            subpixel_scale = int(self.scale_var.get())
            hp_window      = int(self.hp_var.get())
            norm_div       = float(self.norm_var.get())
            if subpixel_scale < 1:
                raise ValueError("Subpixel scale must be ≥ 1")
        except ValueError as exc:
            messagebox.showerror("Invalid parameter", str(exc))
            return

        self.progress.start()
        try:
            # ── 1. Check events.h5 exists ────────────────────────────────────
            h5_path = events_path / "events.h5"
            if not h5_path.exists():
                messagebox.showerror(
                    "events.h5 not found",
                    "Use the 'Decode → .h5' button first to convert the .raw file.",
                )
                return

            # ── 2. Load events ───────────────────────────────────────────────
            self._status("Loading events from HDF5…")
            t, x, y, polarity = load_events(h5_path)

            # ── 3. Resolve velocity ──────────────────────────────────────────
            vx_str = self.vx_var.get().strip()
            if not vx_str:
                messagebox.showerror(
                    "Velocity required",
                    "Enter a vx value or use 'Scan vx' to find it first.",
                )
                return
            vx = float(vx_str)

            # ── 4. TWE reconstruction ────────────────────────────────────────
            self._status(f"Applying TWE  (vx = {vx:.4f} px/s)…")
            imageHP = run_twe(
                t, x, y, polarity,
                vx=vx,
                subpixel_scale=subpixel_scale,
                hp_window=hp_window,
                norm_div=norm_div,
            )

            # Save raw float32 array for downstream quality analysis (results.py).
            np.save(events_path / "reconstructed_raw.npy", imageHP)

            self._status(f"Done.   vx = {vx:.4f} px/s")

            # ── 5. Save + display ────────────────────────────────────────────
            output_path = events_path / "reconstructed_image.png"
            ds = max(1, subpixel_scale // 10)  # downsample for display only

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(-imageHP[:, ::ds], cmap="gray", aspect="auto")
            ax.set_title(f"TWE Reconstruction  ·  vx = {vx:.4f} px/s", fontsize=12)
            ax.set_xlabel(f"Subpixel column  (downsampled ×{ds} for display)")
            ax.set_ylabel("Row")

            fig.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {output_path}")
            plt.show()

        except Exception as exc:
            messagebox.showerror("TWE failed", str(exc))
            self._status(f"Error: {exc}")
        finally:
            self.progress.stop()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = TWEApp()
    app.mainloop()
