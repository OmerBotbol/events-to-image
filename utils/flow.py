import numpy as np
from scipy.optimize import minimize_scalar


def estimate_vx(t, x, y, vx_initial, min_pixel_travel=50, search_range=0.5):
    """
    Estimate horizontal velocity (vx) using a short time window of events.

    Assumes constant speed, so a small window is representative of the full recording.
    Uses a 1D bounded search — much faster than the 2D Nelder-Mead approach.

    The window duration is derived from vx_initial so that the object always travels
    at least min_pixel_travel pixels within the window. This ensures the variance
    landscape has a measurable peak regardless of how fast or slow the object moves.

    The histogram is built over the active pixel region of the window events rather
    than the full camera frame, so far/small objects produce an equally sharp
    variance landscape as close/large ones.

    Args:
        t:                 Timestamps in seconds (full recording).
        x:                 X pixel coordinates (full recording).
        y:                 Y pixel coordinates (full recording).
        vx_initial:        Initial velocity estimate in pixels/s.
        min_pixel_travel:  Minimum pixels the object must travel within the window (default 50).
        search_range:      Fractional search range around vx_initial, e.g. 0.5 = ±50%.

    Returns:
        vx_optimal: Refined horizontal velocity in pixels/s.
    """
    window_s = min_pixel_travel / abs(vx_initial)
    window_s = min(window_s, t[-1] - t[0])

    t_mid = (t[0] + t[-1]) / 2.0
    half  = window_s / 2.0
    mask  = (t >= t_mid - half) & (t <= t_mid + half)

    t_win = t[mask]
    x_win = x[mask]
    y_win = y[mask].astype(np.float64)

    if len(t_win) == 0:
        raise ValueError(
            f"No events found in {window_s*1000:.0f} ms window around t={t_mid:.3f} s"
        )

    t_ref = t_win[-1]

    # Histogram over the active pixel region so far/small objects are not penalised
    # by a mostly-empty full-frame histogram.
    x_lo, x_hi = float(x_win.min()), float(x_win.max())
    y_lo, y_hi = float(y_win.min()), float(y_win.max())
    x_bins = max(int(x_hi - x_lo), 1)
    y_bins = max(int(y_hi - y_lo), 1)

    def neg_variance(vx):
        x_warped = x_win - (t_win - t_ref) * vx
        iwe, _, _ = np.histogram2d(
            x_warped, y_win,
            bins=[x_bins, y_bins],
            range=[[x_lo, x_hi], [y_lo, y_hi]],
        )
        return -np.var(iwe)

    vx_lo = vx_initial * (1 - search_range)
    vx_hi = vx_initial * (1 + search_range)
    result = minimize_scalar(neg_variance, bounds=(vx_lo, vx_hi), method="bounded")
    return float(result.x)
