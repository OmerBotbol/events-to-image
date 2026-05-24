"""
IWE — Image of Warped Events
Reconstructs an image by warping events to a reference time using optical
flow, then accumulating them via bilinear splatting.

Reference:
  Gallego et al., "A Unifying Contrast Maximization Framework for Event
  Cameras, with Applications to the Calibration of a Neuromorphic Camera",
  CVPR 2018.

Usage:  python iwe.py
"""
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from tiwe import _load_events_for_scan
from utils.flow import scan_vx


# ── core IWE computation — streams from HDF5, never loads all events ──────────

def compute_iwe(h5_path, vx: float, vy: float,
                height: int, width: int,
                pol_mode: str = "signed",
                chunk_size: int = 500_000,
                status_cb=None):
    """
    Stream events from HDF5 in chunks and accumulate via bilinear splatting.
    Peak RAM is O(chunk_size) — safe even for recordings with hundreds of
    millions of events.

    Parameters
    ----------
    h5_path    : path to events.h5
    vx, vy     : optical flow in px/s
    height, width : output image dimensions
    pol_mode   : "signed" | "unsigned" | "split"
    chunk_size : events read per HDF5 slice
    status_cb  : optional callable(str) for progress messages

    Returns
    -------
    "signed"/"unsigned" → iwe          (H×W float32)
    "split"             → (iwe_pos, iwe_neg, iwe_combined), each H×W float32
    """
    N        = height * width
    iwe_flat = np.zeros(N, dtype=np.float32)
    neg_flat = np.zeros(N, dtype=np.float32) if pol_mode == "split" else None

    with h5py.File(h5_path, "r") as f:
        evs      = f["events"]
        n_events = len(evs["timestamp"])
        t_ref    = float(evs["timestamp"][n_events - 1])   # one scalar read

        for start in range(0, n_events, chunk_size):
            end = min(start + chunk_size, n_events)

            if status_cb and (start // chunk_size) % 5 == 0:
                pct = 100 * start // n_events
                status_cb(f"Computing IWE… {start:,}/{n_events:,}  ({pct}%)")

            t_c = evs["timestamp"][start:end].astype(np.float32)
            x_c = evs["x"][start:end].astype(np.float32)
            y_c = evs["y"][start:end].astype(np.float32)
            p_c = evs["polarity"][start:end].astype(np.float32)

            dt  = np.float32(t_ref) - t_c;           del t_c
            x_w = x_c + np.float32(vx) * dt;         del x_c
            y_w = y_c + np.float32(vy) * dt;         del y_c, dt

            x0 = np.floor(x_w).astype(np.int32)
            y0 = np.floor(y_w).astype(np.int32)
            wx = (x_w - x0).astype(np.float32)
            wy = (y_w - y0).astype(np.float32)
            del x_w, y_w

            for xi, yi, wi in [
                (x0,     y0,     (1.0 - wx) * (1.0 - wy)),
                (x0 + 1, y0,     wx         * (1.0 - wy)),
                (x0,     y0 + 1, (1.0 - wx) * wy        ),
                (x0 + 1, y0 + 1, wx         * wy        ),
            ]:
                in_b = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
                if not in_b.any():
                    continue
                idx = (yi[in_b] * width + xi[in_b]).astype(np.int32)
                w   = (wi * p_c)[in_b]

                if pol_mode == "unsigned":
                    w = np.abs(w)

                if pol_mode == "split":
                    pos = w > 0
                    neg = w < 0
                    if pos.any():
                        iwe_flat += np.bincount(
                            idx[pos], weights=w[pos], minlength=N
                        ).astype(np.float32)
                    if neg.any():
                        neg_flat += np.bincount(
                            idx[neg], weights=-w[neg], minlength=N
                        ).astype(np.float32)
                else:
                    iwe_flat += np.bincount(
                        idx, weights=w, minlength=N
                    ).astype(np.float32)

    if pol_mode == "split":
        iwe_pos = iwe_flat.reshape(height, width)
        iwe_neg = neg_flat.reshape(height, width)
        return iwe_pos, iwe_neg, iwe_pos - iwe_neg
    return iwe_flat.reshape(height, width)



# ── GUI ───────────────────────────────────────────────────────────────────────

class IWEApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("IWE — Image of Warped Events")
        self.resizable(False, False)
        self._build_ui()

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        P = {"padx": 8, "pady": 4}

        # ── folder row ──
        tk.Label(self, text="Events folder (.h5 file):").grid(
            row=0, column=0, sticky="w", **P)
        self.folder_var = tk.StringVar()
        tk.Entry(self, textvariable=self.folder_var, width=52).grid(
            row=0, column=1, **P)
        tk.Button(self, text="Browse…", command=self._browse).grid(
            row=0, column=2, **P)

        ttk.Separator(self, orient="horizontal").grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=6)

        # ── vx ──
        tk.Label(self, text="Velocity  vx  (px/s):").grid(
            row=2, column=0, sticky="w", **P)
        self.vx_var = tk.StringVar(value="")
        tk.Entry(self, textvariable=self.vx_var, width=15).grid(
            row=2, column=1, sticky="w", **P)
        tk.Label(self, text="Enter manually or use 'Scan vx' below",
                 foreground="gray").grid(row=3, column=1, sticky="w", padx=8)

        # ── vx scan ──
        scan_frame = tk.Frame(self)
        scan_frame.grid(row=4, column=0, columnspan=3, sticky="w", padx=8, pady=2)
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

        # ── vy ──
        tk.Label(self, text="Velocity  vy  (px/s):").grid(
            row=5, column=0, sticky="w", **P)
        self.vy_var = tk.StringVar(value="0.0")
        tk.Entry(self, textvariable=self.vy_var, width=15).grid(
            row=5, column=1, sticky="w", **P)
        tk.Label(self, text="0 for purely horizontal motion",
                 foreground="gray").grid(row=6, column=1, sticky="w", padx=8)

        # ── camera size ──
        size_frame = tk.Frame(self)
        size_frame.grid(row=7, column=0, columnspan=3, sticky="w", padx=8, pady=4)
        tk.Label(size_frame, text="Camera size (W × H):").pack(side="left")
        self.width_var  = tk.StringVar(value="1280")
        self.height_var = tk.StringVar(value="720")
        tk.Entry(size_frame, textvariable=self.width_var,  width=6).pack(side="left", padx=4)
        tk.Label(size_frame, text="×").pack(side="left")
        tk.Entry(size_frame, textvariable=self.height_var, width=6).pack(side="left", padx=4)

        # ── polarity mode ──
        tk.Label(self, text="Polarity mode:").grid(row=8, column=0, sticky="w", **P)
        self.pol_mode_var = tk.StringVar(value="signed")
        pol_frame = tk.Frame(self)
        pol_frame.grid(row=8, column=1, sticky="w", padx=8)
        for label, val in [("Signed (±1)", "signed"),
                            ("Count all", "unsigned"),
                            ("Split ±", "split")]:
            tk.Radiobutton(pol_frame, text=label,
                           variable=self.pol_mode_var, value=val).pack(side="left")

        ttk.Separator(self, orient="horizontal").grid(
            row=9, column=0, columnspan=3, sticky="ew", pady=6)

        # ── status + progress ──
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(
            self, textvariable=self.status_var,
            foreground="blue", wraplength=460, anchor="w",
        ).grid(row=10, column=0, columnspan=3, sticky="w", **P)

        self.progress = ttk.Progressbar(self, mode="indeterminate", length=460)
        self.progress.grid(row=11, column=0, columnspan=3, padx=8, pady=4)

        # ── run button ──
        tk.Button(
            self, text="▶  Compute IWE", command=self._run,
            bg="#4CAF50", fg="white", font=("Arial", 11, "bold"),
            padx=20, pady=8,
        ).grid(row=12, column=0, columnspan=3, pady=14)

    # ── callbacks ────────────────────────────────────────────────────────────

    def _browse(self):
        folder = filedialog.askdirectory(title="Select folder containing events.h5")
        if folder:
            self.folder_var.set(folder)

    def _status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def _h5_path(self) -> Path:
        return Path(self.folder_var.get().strip()) / "events.h5"

    def _scan_vx(self):
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showerror("Error", "Select a folder first.")
            return
        h5 = self._h5_path()
        if not h5.exists():
            messagebox.showerror("Error", "events.h5 not found in that folder.")
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
            t, x, y, _ = _load_events_for_scan(h5, vx_max)
            n_loaded = len(t)

            best_vx, candidates, variances = scan_vx(
                t, x, y, vx_min, vx_max, n_steps,
                status_cb=self._status,
            )
            del t, x, y

            self.vx_var.set(f"{best_vx:.4f}")
            self._status(
                f"Scan complete.  Best vx = {best_vx:.4f} px/s  "
                f"({n_loaded:,} events loaded)"
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(candidates, variances, linewidth=1)
            ax.axvline(best_vx, color="r", linestyle="--",
                       label=f"Peak  {best_vx:.2f} px/s")
            ax.set_xlabel("Candidate  vx  (px/s)")
            ax.set_ylabel("IWE variance  (contrast)")
            ax.set_title("IWE velocity scan — contrast maximisation")
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
        h5_path = events_path / "events.h5"
        if not h5_path.exists():
            messagebox.showerror("Error", "events.h5 not found in that folder.")
            return

        try:
            vx_str = self.vx_var.get().strip()
            if not vx_str:
                messagebox.showerror("Velocity required",
                                     "Enter a vx value or use 'Scan vx' first.")
                return
            vx = float(vx_str)
            vy = float(self.vy_var.get())
        except ValueError:
            messagebox.showerror("Invalid parameter", "Check vx / vy values.")
            return

        try:
            width  = int(self.width_var.get())
            height = int(self.height_var.get())
            if width < 1 or height < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid camera size",
                                 "Enter valid width and height (e.g. 1280 × 720).")
            return

        pol_mode = self.pol_mode_var.get()

        self.progress.start()
        try:
            self._status(f"Computing IWE  (vx={vx:.4f}, vy={vy:.4f})  "
                         f"streaming from HDF5…")

            result = compute_iwe(
                h5_path, vx, vy, height, width,
                pol_mode=pol_mode,
                status_cb=self._status,
            )

            # ── Save ────────────────────────────────────────────────────────
            npy_path = events_path / "iwe_reconstructed.npy"
            if pol_mode == "split":
                iwe_pos, iwe_neg, iwe = result
                np.save(npy_path, iwe)
            else:
                iwe = result
                np.save(npy_path, iwe)

            # ── Display ──────────────────────────────────────────────────────
            self._status(f"Done.  IWE shape: {iwe.shape[1]} × {iwe.shape[0]} px")

            if pol_mode == "split":
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(-iwe_pos, cmap="gray", aspect="auto")
                axes[0].set_title("IWE+  (positive events)")
                axes[1].imshow(iwe_neg,  cmap="gray", aspect="auto")
                axes[1].set_title("IWE−  (negative events)")
                axes[2].imshow(-iwe,     cmap="gray", aspect="auto")
                axes[2].set_title("IWE  (combined)")
                for ax in axes:
                    ax.axis("off")
                fig.suptitle(f"IWE  ·  vx={vx:.4f} px/s,  vy={vy:.4f} px/s",
                             fontsize=12)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(-iwe, cmap="gray", aspect="auto")
                ax.set_title(f"IWE  ·  vx={vx:.4f} px/s,  vy={vy:.4f} px/s",
                             fontsize=12)
                ax.axis("off")

            fig.tight_layout()
            png_path = events_path / "iwe_reconstructed.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {npy_path}")
            print(f"Saved → {png_path}")
            plt.show()

        except Exception as exc:
            messagebox.showerror("IWE failed", str(exc))
            self._status(f"Error: {exc}")
        finally:
            self.progress.stop()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = IWEApp()
    app.mainloop()
