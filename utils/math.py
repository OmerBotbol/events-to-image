import numpy as np


def calculate_object_velocity(
    duration_s: float,
    camera_resolution_px: tuple[int, int],
    roi_x_start_px: int = None,
    roi_x_end_px: int = None,
) -> float:
    """
    Calculate the linear velocity of an object moving across a screen in pixels/s.

    Args:
        duration_s:            Time the object takes to travel from start to end (seconds).
        camera_resolution_px:  Camera resolution as (width_px, height_px).
        roi_x_start_px:        Pixel x where movement starts. Defaults to 0.
        roi_x_end_px:          Pixel x where movement ends. Defaults to frame width.

    Returns:
        Velocity in pixels/s.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    cam_width_px, _ = camera_resolution_px
    if roi_x_start_px is None:
        roi_x_start_px = 0
    if roi_x_end_px is None:
        roi_x_end_px = cam_width_px

    if not (0 <= roi_x_start_px <= cam_width_px and 0 <= roi_x_end_px <= cam_width_px):
        raise ValueError("ROI pixel coordinates must be within camera resolution")

    return abs(roi_x_end_px - roi_x_start_px) / duration_s


def compute_cpp(image: np.ndarray, subpixel_scale: float = 100.0) -> dict:
    """
    Compute the dominant spatial frequency (cycles per pixel) of the reconstructed image.

    Uses the 1D power spectrum along the x axis (axis=1), averaged across all rows,
    then picks the frequency bin with the highest power (DC excluded).

    Args:
        image:           2D reconstructed image array, shape (rows, cols).
        subpixel_scale:  The scale factor applied when building pix during reconstruction
                         (default 100, matching `round(x * 1e2)`). The image is averaged
                         in groups of this size before the FFT, reducing it to camera-pixel
                         resolution and avoiding large memory allocations.

    Returns:
        dict with:
            cpp_camera_px:   Dominant frequency in cycles per camera pixel.
            frequency_axis:  Frequency array in cycles per camera pixel (0 to 0.5).
            power_spectrum:  Mean power spectrum across rows at camera-pixel resolution.
    """
    scale = int(subpixel_scale)

    # Keep 10 subpixels per camera pixel so the FFT can see up to 5.0 CPP.
    # Downsampling all the way to 1 px/camera-px would hard-cap the result at 0.5 CPP,
    # making super-resolution detection impossible.
    ds = max(1, scale // 10)          # subpixels to average per output sample
    effective_scale = scale / ds      # output samples per camera pixel  (= 10 when scale=100)

    n_trim = (image.shape[1] // ds) * ds
    image_ds = image[:, :n_trim].reshape(image.shape[0], -1, ds).mean(axis=2)

    N = image_ds.shape[1]
    fft_mag = np.abs(np.fft.rfft(image_ds, axis=1))
    power = np.mean(fft_mag ** 2, axis=0)

    # freq in cycles/output-sample → convert to cycles/camera-pixel (CPP)
    freq_axis = np.fft.rfftfreq(N) * effective_scale

    # Exclude DC (index 0) when searching for the dominant peak
    peak_idx = int(np.argmax(power[1:]) + 1)
    cpp_camera_px = float(freq_axis[peak_idx])

    return {
        "cpp_camera_px":  cpp_camera_px,
        "frequency_axis": freq_axis,
        "power_spectrum": power,
    }