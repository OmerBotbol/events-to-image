from utils import decoder
from pathlib import Path
import h5py

  
script_dir = Path(__file__).parent
events_path = script_dir / "events" / "experiment_0" # change the last string to the events' folder

## First we need to convert the events into a .hdf5 file, uncommend the next line to do so and make sure that the events data seats in the "events" folder
# decoder.runDecoder(events_path)
decoder.convertToHDF5(script_dir / "csv" / "out0.csv",events_path)

hdf5_path = events_path / "events.h5"

events = h5py.File(hdf5_path, "r")
print(events.keys())