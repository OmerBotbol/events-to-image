from utils import decoder
from utils.math import calculate_object_velocity
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

script_dir = Path(__file__).parent
events_path = script_dir / "events" / "experiment_10"  # change to the events folder

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

# --- Compute vx (pixels/s) from the physical setup ---
SCREEN_WIDTH_M  = 0.34
CAM_WIDTH_PX    = 1280
CAM_HEIGHT_PX   = 720
CAMERA_DIST_M   = 0.32

vel = calculate_object_velocity(
    screen_width_m=SCREEN_WIDTH_M,
    camera_distance_m=CAMERA_DIST_M,
    duration_s=34.7,
    camera_resolution_px=(CAM_WIDTH_PX, CAM_HEIGHT_PX),
    roi_x_start_px=510  ,
    roi_x_end_px=620,
)
vx = vel["linear_velocity_m_s"] * (CAM_WIDTH_PX / SCREEN_WIDTH_M)  # pixels/s
# vx-=1
print(vx)


# --- Velocity compensation: warp each event's x by removing bulk motion ---
# x_warped = x - vx * t  (equivalent to MATLAB: events_new(:,2) - vx.*events_new(:,1))
x_warped = x - vx * t

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

# --- Display and save ---
fig, ax = plt.subplots()
ax.imshow(-imageHP[:, ::100], cmap="gray", aspect="auto")
ax.set_title("High-pass filtered event image")
fig.colorbar(ax.images[0], ax=ax)

output_path = events_path / "reconstructed_image.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Image saved to {output_path}")

plt.show()
