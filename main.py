from utils import decoder
from utils.math import calculate_object_velocity, compute_cpp
from utils.flow import estimate_vx
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

script_dir = Path(__file__).parent
events_path = script_dir / "events" / "experiment_0"  # change to the events folder

## Uncomment to decode raw files and convert to HDF5 first:
# decoder.runDecoder(events_path)
# decoder.convertToHDF5(script_dir / "csv" / "out10.csv", events_path)

hdf5_path = events_path / "events.h5"

with h5py.File(hdf5_path, "r") as f:
    evs = f["events"]
    t        = evs["timestamp"][:]          # seconds
    x        = evs["x"][:].astype(np.float64)
    y        = evs["y"][:].astype(np.int32)
    polarity = evs["polarity"][:].astype(np.float64)  # -1 or +1

# --- Compute vx initial estimate (pixels/s) from the physical setup ---
CAM_WIDTH_PX  = 1280
CAM_HEIGHT_PX = 720

vx_initial = calculate_object_velocity(
    duration_s=7,
    camera_resolution_px=(CAM_WIDTH_PX, CAM_HEIGHT_PX),
    roi_x_start_px=240,
    roi_x_end_px=1222,
)

# --- Estimate vx via optical flow (1D contrast maximization on a 75 ms window) ---
vx = estimate_vx(t, x, y, vx_initial=vx_initial)
print(f"vx: {vx:.4f} px/s  (initial estimate: {vx_initial:.4f} px/s)")

# --- Velocity compensation: warp events to the last timestamp as reference ---
t_ref = t[-1]
x_warped = x - (t - t_ref) * vx

# --- Map warped x positions to integer pixel indices at 100x sub-pixel resolution ---
# MATLAB: pix = round(x_indices*1e2) - min(round(x_indices*1e2)) + 1
pix = np.round(x_warped * 1e2).astype(np.int64)
pix -= pix.min()  # shift to start at 0

# --- Accumulate polarities into a 2D image (sum over events sharing same y, pix) ---
# MATLAB: vq = accumarray([y_indices+1, pix], events_new(:,4))
num_rows = int(y.max()) + 1
num_cols = int(pix.max()) + 1
image = coo_matrix((polarity, (y, pix)), shape=(num_rows, num_cols)).toarray()

# --- Cumulative sum along the pixel axis ---
# MATLAB: image = cumsum(image, 2)
image = np.cumsum(image, axis=1)

# --- Normalise ---
# MATLAB: image = image / 65
image = image / 65

# --- High-pass filter: subtract local moving mean along pixel axis ---
# MATLAB: imageHP = image - movmean(image', 1500)'
moving_avg = uniform_filter1d(image, size=1500, axis=1, mode="nearest")
imageHP = image - moving_avg

# --- Compute CPP of the reconstructed image ---
cpp_result = compute_cpp(imageHP, subpixel_scale=100.0)
print(f"CPP: {cpp_result['cpp_camera_px']:.4f} cycles/camera-pixel")

# --- Display every 100th column, inverted ---
# MATLAB: figure; imshow(-imageHP(:,1:100:end)*1)
plt.figure()
# plt.imshow(-image[:, ::100], cmap="gray", aspect="auto")
plt.imshow(-imageHP[:, ::100], cmap="gray", aspect="auto")
plt.title("High-pass filtered event image")
plt.colorbar()
plt.show()
