"""
compare.py — Reconstruct a brightness image from an IWE using L2, L1, and CNN
regularization, then display and save a side-by-side comparison figure.

Workflow
--------
1. Select the events folder (must contain iwe_reconstructed.npy saved by iwe.py,
   or run iwe.py first to generate it).
2. Enter vx (and optionally vy) — same values used in iwe.py.
3. Adjust regularization weights if desired.
4. Click individual method buttons or "Run All & Compare".

CNN requires drunet_gray.pth in the models/ folder.
Download: https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth
"""
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils.reconstruct import reconstruct_l2, reconstruct_l1, reconstruct_cnn
from tiwe import load_events
from utils.flow import scan_vx


# ── app ───────────────────────────────────────────────────────────────────────

class CompareApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Reconstruction Comparison — L2 / L1 / CNN")
        self.resizable(False, False)
        self._results: dict[str, np.ndarray] = {}
        self._iwe: np.ndarray | None = None
        self._build_ui()

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        P = {"padx": 8, "pady": 4}

        # ── folder ──
        tk.Label(self, text="Events folder (contains iwe_reconstructed.npy):").grid(
            row=0, column=0, sticky="w", **P)
        self.folder_var = tk.StringVar()
        tk.Entry(self, textvariable=self.folder_var, width=50).grid(
            row=0, column=1, **P)
        tk.Button(self, text="Browse…", command=self._browse).grid(
            row=0, column=2, **P)

        ttk.Separator(self, orient="horizontal").grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=6)

        # ── flow ──
        flow_frame = tk.Frame(self)
        flow_frame.grid(row=2, column=0, columnspan=3, sticky="w", padx=8)
        tk.Label(flow_frame, text="vx (px/s):").pack(side="left")
        self.vx_var = tk.StringVar(value="")
        tk.Entry(flow_frame, textvariable=self.vx_var, width=10).pack(
            side="left", padx=4)
        tk.Label(flow_frame, text="   vy (px/s):").pack(side="left")
        self.vy_var = tk.StringVar(value="0.0")
        tk.Entry(flow_frame, textvariable=self.vy_var, width=10).pack(
            side="left", padx=4)

        # ── vx scan ──
        scan_frame = tk.Frame(self)
        scan_frame.grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=2)
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

        ttk.Separator(self, orient="horizontal").grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=6)

        # ── L2 params ──
        l2_frame = tk.LabelFrame(self, text="L2 — Tikhonov", padx=6, pady=4)
        l2_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=8, pady=2)
        tk.Label(l2_frame, text="Regularisation weight λ:").grid(row=0, column=0, sticky="w")
        self.l2_weight_var = tk.StringVar(value="0.3")
        tk.Entry(l2_frame, textvariable=self.l2_weight_var, width=8).grid(
            row=0, column=1, sticky="w", padx=4)
        tk.Label(l2_frame, text="  LSQR iterations:").grid(row=0, column=2, sticky="w")
        self.l2_iter_var = tk.StringVar(value="100")
        tk.Entry(l2_frame, textvariable=self.l2_iter_var, width=6).grid(
            row=0, column=3, sticky="w", padx=4)
        tk.Label(l2_frame, text="  HP filter window (0=off):").grid(row=0, column=4, sticky="w")
        self.l2_hp_var = tk.StringVar(value="200")
        tk.Entry(l2_frame, textvariable=self.l2_hp_var, width=6).grid(
            row=0, column=5, sticky="w", padx=4)
        tk.Button(l2_frame, text="▶ Run L2", command=lambda: self._run_method("l2"),
                  bg="#1565C0", fg="white", font=("Arial", 9, "bold")).grid(
            row=0, column=6, padx=10)

        # ── L1 params ──
        l1_frame = tk.LabelFrame(self, text="L1 — Total Variation", padx=6, pady=4)
        l1_frame.grid(row=6, column=0, columnspan=3, sticky="ew", padx=8, pady=2)
        tk.Label(l1_frame, text="Regularisation weight λ:").grid(row=0, column=0, sticky="w")
        self.l1_weight_var = tk.StringVar(value="0.1")
        tk.Entry(l1_frame, textvariable=self.l1_weight_var, width=8).grid(
            row=0, column=1, sticky="w", padx=4)
        tk.Label(l1_frame, text="  Outer iters:").grid(row=0, column=2, sticky="w")
        self.l1_outer_var = tk.StringVar(value="20")
        tk.Entry(l1_frame, textvariable=self.l1_outer_var, width=5).grid(
            row=0, column=3, sticky="w", padx=4)
        tk.Label(l1_frame, text="  Inner iters:").grid(row=0, column=4, sticky="w")
        self.l1_inner_var = tk.StringVar(value="5")
        tk.Entry(l1_frame, textvariable=self.l1_inner_var, width=5).grid(
            row=0, column=5, sticky="w", padx=4)
        tk.Label(l1_frame, text="  HP filter window (0=off):").grid(row=0, column=6, sticky="w")
        self.l1_hp_var = tk.StringVar(value="200")
        tk.Entry(l1_frame, textvariable=self.l1_hp_var, width=6).grid(
            row=0, column=7, sticky="w", padx=4)
        tk.Button(l1_frame, text="▶ Run L1", command=lambda: self._run_method("l1"),
                  bg="#2E7D32", fg="white", font=("Arial", 9, "bold")).grid(
            row=0, column=8, padx=10)

        # ── CNN params ──
        cnn_frame = tk.LabelFrame(self, text="CNN — HQS + DRUNet", padx=6, pady=4)
        cnn_frame.grid(row=7, column=0, columnspan=3, sticky="ew", padx=8, pady=2)
        tk.Label(cnn_frame, text="Model path:").grid(row=0, column=0, sticky="w")
        self.cnn_model_var = tk.StringVar(value="models/drunet_gray.pth")
        tk.Entry(cnn_frame, textvariable=self.cnn_model_var, width=30).grid(
            row=0, column=1, sticky="w", padx=4)
        tk.Button(cnn_frame, text="…", command=self._browse_model, width=2).grid(
            row=0, column=2)
        tk.Label(cnn_frame, text="  HQS iters:").grid(row=0, column=3, sticky="w")
        self.cnn_hqs_var = tk.StringVar(value="16")
        tk.Entry(cnn_frame, textvariable=self.cnn_hqs_var, width=5).grid(
            row=0, column=4, sticky="w", padx=4)
        tk.Label(cnn_frame, text="  Grad iters:").grid(row=0, column=5, sticky="w")
        self.cnn_grad_var = tk.StringVar(value="100")
        tk.Entry(cnn_frame, textvariable=self.cnn_grad_var, width=6).grid(
            row=0, column=6, sticky="w", padx=4)
        tk.Button(cnn_frame, text="▶ Run CNN", command=lambda: self._run_method("cnn"),
                  bg="#6A1B9A", fg="white", font=("Arial", 9, "bold")).grid(
            row=0, column=7, padx=10)
        tk.Label(cnn_frame,
                 text="Requires drunet_gray.pth — download from github.com/cszn/KAIR/releases",
                 foreground="gray").grid(row=1, column=0, columnspan=8, sticky="w")

        ttk.Separator(self, orient="horizontal").grid(
            row=8, column=0, columnspan=3, sticky="ew", pady=6)

        # ── compare button ──
        tk.Button(
            self, text="▶  Run All & Compare",
            command=self._run_all,
            bg="#E65100", fg="white", font=("Arial", 11, "bold"),
            padx=20, pady=8,
        ).grid(row=9, column=0, columnspan=3, pady=6)

        # ── status + progress ──
        self.status_var = tk.StringVar(value="Ready.  Load a folder and enter vx.")
        tk.Label(self, textvariable=self.status_var,
                 foreground="blue", wraplength=560, anchor="w").grid(
            row=10, column=0, columnspan=3, sticky="w", **P)

        self.progress = ttk.Progressbar(self, mode="indeterminate", length=560)
        self.progress.grid(row=11, column=0, columnspan=3, padx=8, pady=4)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _browse(self):
        folder = filedialog.askdirectory(title="Select events folder")
        if folder:
            self.folder_var.set(folder)

    def _scan_vx(self):
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showerror("Error", "Select a folder first.")
            return
        h5 = Path(folder) / "events.h5"
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
            t, x, y, _ = load_events(h5)
            n_total = len(t)

            best_vx, candidates, variances = scan_vx(
                t, x, y, vx_min, vx_max, n_steps,
                status_cb=self._status,
            )
            del t, x, y

            self.vx_var.set(f"{best_vx:.4f}")
            self._status(
                f"Scan complete.  Best vx = {best_vx:.4f} px/s  "
                f"({n_total:,} events total)"
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(candidates, variances, linewidth=1)
            ax.axvline(best_vx, color="r", linestyle="--",
                       label=f"Peak  {best_vx:.2f} px/s")
            ax.set_xlabel("Candidate  vx  (px/s)")
            ax.set_ylabel("IWE variance  (contrast)")
            ax.set_title("Velocity scan — contrast maximisation")
            ax.legend()
            fig.tight_layout()
            plt.show()

        except Exception as exc:
            messagebox.showerror("Scan failed", str(exc))
            self._status(f"Scan error: {exc}")
        finally:
            self.progress.stop()

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select drunet_gray.pth",
            filetypes=[("PyTorch weights", "*.pth"), ("All files", "*.*")],
        )
        if path:
            self.cnn_model_var.set(path)

    def _status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def _load_iwe(self) -> np.ndarray | None:
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showerror("Error", "Select a folder first.")
            return None
        npy = Path(folder) / "iwe_reconstructed.npy"
        if not npy.exists():
            messagebox.showerror(
                "IWE not found",
                "iwe_reconstructed.npy not found in that folder.\n"
                "Run iwe.py first to compute and save the IWE.",
            )
            return None
        self._status("Loading IWE…")
        return np.load(str(npy))

    def _parse_flow(self):
        try:
            vx = float(self.vx_var.get())
            vy = float(self.vy_var.get())
            return vx, vy
        except ValueError:
            messagebox.showerror("Invalid flow", "vx and vy must be numbers.")
            return None

    def _save_and_show(self):
        """Display all available results and save comparison PNG."""
        results = self._results
        if not results:
            return

        iwe = self._iwe
        keys   = list(results.keys())
        titles = {"l2": "L2 — Tikhonov", "l1": "L1 — Total Variation",
                  "cnn": "CNN — HQS + DRUNet"}

        n = len(keys) + 1  # +1 for IWE
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        # IWE panel
        axes[0].imshow(-iwe, cmap="gray", aspect="auto")
        axes[0].set_title("IWE (input)", fontsize=11)
        axes[0].axis("off")

        for ax, key in zip(axes[1:], keys):
            ax.imshow(results[key], cmap="gray", aspect="auto")
            ax.set_title(titles.get(key, key), fontsize=11)
            ax.axis("off")

        fig.suptitle("Event-based image reconstruction — regularisation comparison",
                     fontsize=12)
        fig.tight_layout()

        folder = Path(self.folder_var.get().strip())
        out_png = folder / "reconstruction_comparison.png"
        fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
        print(f"Saved → {out_png}")

        # Also save individual .npy files
        for key, img in results.items():
            np.save(str(folder / f"reconstructed_{key}.npy"), img)
            print(f"Saved → {folder / f'reconstructed_{key}.npy'}")

        self._status(f"Done. Comparison saved to {out_png.name}")
        plt.show()

    # ── run methods ───────────────────────────────────────────────────────────

    def _run_method(self, method: str):
        iwe = self._load_iwe()
        if iwe is None:
            return
        flow = self._parse_flow()
        if flow is None:
            return
        vx, vy = flow
        self._iwe = iwe

        self.progress.start()
        try:
            if method == "l2":
                lam  = float(self.l2_weight_var.get())
                itr  = int(self.l2_iter_var.get())
                hp   = int(self.l2_hp_var.get())
                result = reconstruct_l2(iwe, vx, vy, reg_weight=lam,
                                        iter_lim=itr, hp_window=hp,
                                        status_cb=self._status)
            elif method == "l1":
                lam   = float(self.l1_weight_var.get())
                outer = int(self.l1_outer_var.get())
                inner = int(self.l1_inner_var.get())
                hp    = int(self.l1_hp_var.get())
                result = reconstruct_l1(iwe, vx, vy, reg_weight=lam,
                                        niter_outer=outer, niter_inner=inner,
                                        hp_window=hp, status_cb=self._status)
            elif method == "cnn":
                hqs  = int(self.cnn_hqs_var.get())
                grad = int(self.cnn_grad_var.get())
                result = reconstruct_cnn(iwe, vx, vy,
                                         model_path=self.cnn_model_var.get().strip(),
                                         n_hqs_iters=hqs, n_grad_iters=grad,
                                         status_cb=self._status)

            self._results[method] = result
            self._save_and_show()

        except FileNotFoundError as exc:
            messagebox.showerror("File not found", str(exc))
            self._status(f"Error: {exc}")
        except Exception as exc:
            import traceback; traceback.print_exc()
            messagebox.showerror(f"{method.upper()} failed", str(exc))
            self._status(f"Error: {exc}")
        finally:
            self.progress.stop()

    def _run_all(self):
        iwe = self._load_iwe()
        if iwe is None:
            return
        flow = self._parse_flow()
        if flow is None:
            return
        if not self.vx_var.get().strip():
            messagebox.showerror("vx required", "Enter a vx value first.")
            return

        vx, vy = flow
        self._iwe = iwe
        self._results.clear()

        self.progress.start()
        try:
            # L2
            lam = float(self.l2_weight_var.get())
            itr = int(self.l2_iter_var.get())
            hp  = int(self.l2_hp_var.get())
            self._results["l2"] = reconstruct_l2(
                iwe, vx, vy, reg_weight=lam, iter_lim=itr, hp_window=hp,
                status_cb=self._status)

            # L1
            lam   = float(self.l1_weight_var.get())
            outer = int(self.l1_outer_var.get())
            inner = int(self.l1_inner_var.get())
            hp    = int(self.l1_hp_var.get())
            self._results["l1"] = reconstruct_l1(
                iwe, vx, vy, reg_weight=lam, niter_outer=outer, niter_inner=inner,
                hp_window=hp, status_cb=self._status)

            # CNN (optional — skip gracefully if model missing)
            try:
                hqs  = int(self.cnn_hqs_var.get())
                grad = int(self.cnn_grad_var.get())
                self._results["cnn"] = reconstruct_cnn(
                    iwe, vx, vy,
                    model_path=self.cnn_model_var.get().strip(),
                    n_hqs_iters=hqs, n_grad_iters=grad,
                    status_cb=self._status)
            except FileNotFoundError as exc:
                self._status(f"CNN skipped: {exc}")
                messagebox.showwarning(
                    "CNN skipped",
                    "DRUNet model file not found — CNN result will be omitted.\n\n"
                    + str(exc),
                )

            self._save_and_show()

        except Exception as exc:
            import traceback; traceback.print_exc()
            messagebox.showerror("Reconstruction failed", str(exc))
            self._status(f"Error: {exc}")
        finally:
            self.progress.stop()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = CompareApp()
    app.mainloop()
