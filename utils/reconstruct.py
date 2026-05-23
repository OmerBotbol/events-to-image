"""
Image reconstruction from IWE using three regularization strategies:

  L2 — Tikhonov (Laplacian smoothness), fast, soft edges
  L1 — Total Variation (Split Bregman), slower, sharp edges
  CNN — HQS with pretrained DRUNet denoiser prior, best quality

Forward model:  A * ℓ = b
  A  — directional derivative operator: vx·∂/∂x + vy·∂/∂y
  ℓ  — log-brightness image (what we solve for), vectorised
  b  — IWE (Image of Warped Events), vectorised

References
----------
Zhang et al., "Formulating Event-based Image Reconstruction as a Linear
Inverse Problem with Deep Regularization using Optical Flow",
IEEE TPAMI 2022 / TU Berlin event_based_image_rec_inverse_problem.
"""
import numpy as np
import pylops
from scipy.ndimage import uniform_filter1d

from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _minmax_norm(img: np.ndarray) -> np.ndarray:
    lo, hi = img.min(), img.max()
    return np.zeros_like(img) if hi == lo else (img - lo) / (hi - lo)


def _build_op(height: int, width: int, vx: float, vy: float) -> pylops.LinearOperator:
    """
    Directional derivative operator  A = vx·Dx + vy·Dy.
    Acts on a vectorised (height×width) image.
    """
    Dx = pylops.FirstDerivative(dims=(height, width), axis=1,
                                edge=False, dtype=np.float64)
    Dy = pylops.FirstDerivative(dims=(height, width), axis=0,
                                edge=False, dtype=np.float64)
    return float(vx) * Dx + float(vy) * Dy


def _hp_filter(img: np.ndarray, window: int) -> np.ndarray:
    """Subtract moving mean along x-axis to remove horizontal banding."""
    moving_avg = uniform_filter1d(img, size=window, axis=1, mode="nearest")
    return img - moving_avg


def _norm_iwe(iwe: np.ndarray) -> np.ndarray:
    """Normalise IWE to [-1, 1] so it meshes with a [0,1] brightness image."""
    mx = np.abs(iwe).max()
    return iwe.astype(np.float64) / mx if mx > 0 else iwe.astype(np.float64)


# ── L2 — Tikhonov / Laplacian smoothness ─────────────────────────────────────

def reconstruct_l2(iwe: np.ndarray, vx: float, vy: float,
                   reg_weight: float = 0.3,
                   iter_lim: int = 100,
                   hp_window: int = 0,
                   status_cb=None) -> np.ndarray:
    """
    Solve  min_ℓ  ‖A·ℓ − b‖²  +  λ ‖∇²ℓ‖²  (Tikhonov / Laplacian).

    Parameters
    ----------
    iwe        : 2-D float array (height × width)
    vx, vy     : optical flow in px/s
    reg_weight : λ — larger → smoother image
    iter_lim   : max LSQR iterations
    hp_window  : high-pass filter window along x (0 = disabled)
    status_cb  : optional callable(str) for progress messages

    Returns
    -------
    Reconstructed brightness image in [0, 1], shape (height, width).
    """
    if status_cb:
        status_cb("L2: building operator…")
    height, width = iwe.shape
    A = _build_op(height, width, vx, vy)
    b = _norm_iwe(iwe).ravel()

    Laplacian = pylops.Laplacian(dims=(height, width), edge=True, dtype=np.float64)

    if status_cb:
        status_cb("L2: running regularised inversion (LSQR)…")

    x, *_ = pylops.optimization.leastsquares.regularized_inversion(
        A, b, [Laplacian], epsRs=[reg_weight],
        **{"iter_lim": iter_lim},
    )
    img = x.reshape(height, width)
    if hp_window > 1:
        if status_cb:
            status_cb("L2: applying high-pass filter…")
        img = _hp_filter(img, hp_window)
    return _minmax_norm(img)


# ── L1 — Total Variation / Split Bregman ─────────────────────────────────────

def reconstruct_l1(iwe: np.ndarray, vx: float, vy: float,
                   reg_weight: float = 0.1,
                   niter_outer: int = 20,
                   niter_inner: int = 5,
                   hp_window: int = 0,
                   status_cb=None) -> np.ndarray:
    """
    Solve  min_ℓ  ‖A·ℓ − b‖²  +  λ (‖Dy·ℓ‖₁ + ‖Dx·ℓ‖₁)  (Total Variation).

    Parameters
    ----------
    iwe          : 2-D float array (height × width)
    vx, vy       : optical flow in px/s
    reg_weight   : λ — larger → sparser / sharper edges
    niter_outer  : outer Split Bregman iterations
    niter_inner  : inner LSQR iterations per outer step
    hp_window    : high-pass filter window along x (0 = disabled)
    status_cb    : optional callable(str) for progress messages

    Returns
    -------
    Reconstructed brightness image in [0, 1], shape (height, width).
    """
    if status_cb:
        status_cb("L1/TV: building operator…")
    height, width = iwe.shape
    A = _build_op(height, width, vx, vy)
    b = _norm_iwe(iwe).ravel()

    x0 = np.full(height * width, 0.5, dtype=np.float64)

    Dop = [
        pylops.FirstDerivative(dims=(height, width), axis=0,
                               edge=False, dtype=np.float64),
        pylops.FirstDerivative(dims=(height, width), axis=1,
                               edge=False, dtype=np.float64),
    ]

    if status_cb:
        status_cb("L1/TV: running Split Bregman…")

    x, *_ = pylops.optimization.sparsity.splitbregman(
        A, b, Dop,
        x0=x0,
        niter_outer=niter_outer,
        niter_inner=niter_inner,
        mu=1.0,
        epsRL1s=[reg_weight, reg_weight],
        tol=1e-4,
        tau=1.0,
        **{"iter_lim": niter_inner, "damp": 1e-4},
    )
    img = x.reshape(height, width)
    if hp_window > 1:
        if status_cb:
            status_cb("L1/TV: applying high-pass filter…")
        img = _hp_filter(img, hp_window)
    return _minmax_norm(img)


# ── CNN — HQS with DRUNet denoiser prior ─────────────────────────────────────

def reconstruct_cnn(iwe: np.ndarray, vx: float, vy: float,
                    model_path: str = "models/drunet_gray.pth",
                    weight1: float = 2.5,
                    weight2: float = 1.3,
                    n_hqs_iters: int = 16,
                    n_grad_iters: int = 100,
                    lr: float = 0.03,
                    status_cb=None) -> np.ndarray:
    """
    HQS (Half Quadratic Splitting) with DRUNet denoiser prior.

    Alternates between:
      (1) gradient descent  min_ℓ  ‖A·ℓ − b‖²  +  ρ‖ℓ − z‖²
      (2) CNN denoising     z  ←  DRUNet(ℓ, σ)

    Parameters
    ----------
    iwe          : 2-D float array (height × width)
    vx, vy       : optical flow in px/s
    model_path   : path to drunet_gray.pth weights file
    weight1/2    : HQS schedule scale factors (from TU Berlin paper)
    n_hqs_iters  : HQS outer iterations
    n_grad_iters : gradient descent iterations per HQS step
    lr           : Adam learning rate
    status_cb    : optional callable(str) for progress messages

    Returns
    -------
    Reconstructed brightness image in [0, 1], shape (height, width).

    Raises
    ------
    FileNotFoundError if model_path does not exist.
    ImportError      if torch is not installed.
    """
    import torch
    import torch.optim as optim

    from utils.drunet import load_drunet, drunet_denoise

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"DRUNet weights not found at '{model_path}'.\n"
            "Download drunet_gray.pth from:\n"
            "  https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth\n"
            "and place it in the 'models/' folder."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if status_cb:
        status_cb(f"CNN: loading DRUNet on {device}…")

    model = load_drunet(str(model_path), device)

    height, width = iwe.shape
    A = _build_op(height, width, vx, vy)
    b_np = _norm_iwe(iwe).ravel()

    # Initialise from L1 solution (warm start)
    if status_cb:
        status_cb("CNN: warm-start from L1…")
    x_np = reconstruct_l1(iwe, vx, vy, reg_weight=0.07,
                           niter_outer=10, niter_inner=5)

    # HQS noise / rho schedules (log-spaced, from TU Berlin paper)
    sigmas = np.logspace(np.log10(weight1 * 49 + 1),
                         np.log10(weight2 * 49 + 1),
                         n_hqs_iters) / 49.0 * 255.0  # noise level in [0,255]
    rhos   = sigmas ** 2 / 255.0 ** 2 * 0.23          # coupling weight

    # Convert IWE to torch
    b_t = torch.tensor(b_np, dtype=torch.float32, device=device)

    def _A_matvec(v_np):
        """Forward operator A applied to a numpy vector."""
        return A.matvec(v_np.astype(np.float64)).astype(np.float32)

    def _At_matvec(v_np):
        """Adjoint A^T applied to a numpy vector."""
        return A.rmatvec(v_np.astype(np.float64)).astype(np.float32)

    x_np_flat = x_np.ravel().astype(np.float32)

    for i in range(n_hqs_iters):
        rho = float(rhos[i])
        sigma = float(sigmas[i])
        if status_cb:
            status_cb(f"CNN: HQS iteration {i+1}/{n_hqs_iters}  (σ={sigma:.1f})…")

        # ── step 1: gradient descent on the data + proximal term ──────────
        z_np = x_np_flat.copy()   # denoised estimate (initialised as current x)

        x_t = torch.tensor(x_np_flat, dtype=torch.float32,
                            device=device, requires_grad=False)
        x_t = x_t.detach().clone().requires_grad_(True)
        optimizer = optim.Adam([x_t], lr=lr)

        z_t = torch.tensor(z_np, dtype=torch.float32, device=device)

        for _ in range(n_grad_iters):
            optimizer.zero_grad()
            x_cpu = x_t.detach().cpu().numpy()
            Ax    = torch.tensor(_A_matvec(x_cpu), device=device)
            loss  = ((Ax - b_t) ** 2).sum() + rho * ((x_t - z_t) ** 2).sum()
            loss.backward()
            optimizer.step()

        x_np_flat = x_t.detach().cpu().numpy()

        # ── step 2: CNN denoising ──────────────────────────────────────────
        x_img = torch.tensor(x_np_flat.reshape(1, 1, height, width),
                              dtype=torch.float32, device=device)
        # Clamp to [0,1] before denoising
        x_img = (x_img - x_img.min()) / (x_img.max() - x_img.min() + 1e-8)
        x_img = drunet_denoise(model, x_img, sigma, device)
        x_np_flat = x_img.cpu().numpy().ravel()

    return _minmax_norm(x_np_flat.reshape(height, width))
