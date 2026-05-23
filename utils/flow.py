import numpy as np


def scan_vx(t, x, y, vx_min: float, vx_max: float, n_steps: int = 300,
            status_cb=None) -> tuple:
    """
    Scan candidate velocities; return (best_vx, candidates, variances).

    Selects a short time window centred in the recording so that a moving
    object spans ~300 px at vx_max.  Subsamples to 100 k events for speed.
    Uses IWE variance (contrast maximisation) as the focus metric.

    Parameters
    ----------
    t, x, y   : event arrays (timestamp, x-coord, y-coord)
    vx_min/max: scan range in px/s
    n_steps   : number of candidate velocities
    status_cb : optional callable(str) for progress messages

    Returns
    -------
    best_vx   : float
    candidates: np.ndarray  shape (n_steps,)
    variances : np.ndarray  shape (n_steps,)
    """
    total_duration = float(t[-1] - t[0])
    window_s = min(300.0 / max(vx_max, 1.0), total_duration)
    t_mid = (float(t[0]) + float(t[-1])) / 2.0
    mask  = (t >= t_mid - window_s / 2) & (t <= t_mid + window_s / 2)

    if mask.sum() == 0:
        t_w = t.astype(np.float64)
        x_w = x.astype(np.float64)
        y_w = y.astype(np.float64)
        window_s = total_duration
    else:
        t_w = t[mask].astype(np.float64)
        x_w = x[mask].astype(np.float64)
        y_w = y[mask].astype(np.float64)

    MAX_EVENTS = 100_000
    if len(t_w) > MAX_EVENTS:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(t_w), MAX_EVENTS, replace=False))
        t_w, x_w, y_w = t_w[idx], x_w[idx], y_w[idx]

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
        if status_cb is not None and i % 20 == 0:
            status_cb(f"Scanning… {i}/{n_steps}  ({100 * i // n_steps}%)")

    best_vx = float(candidates[np.argmax(variances)])
    return best_vx, candidates, variances
