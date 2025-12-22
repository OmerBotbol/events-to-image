from utils import decoder
from pathlib import Path
import polars as pl

  
script_dir = Path(__file__).parent
events_path = script_dir / "events" / "experiment_0" # change the last string to the events' folder

## First we need to convert the events into a .hdf5 file, uncommend the next line to do so and make sure that the events data seats in the "events" folder
# decoder.runDecoder(events_path)

