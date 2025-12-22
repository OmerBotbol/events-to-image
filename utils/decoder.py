import polars as pl
import numpy as np
from pathlib import Path
import h5py
import subprocess
import os
import glob

def convertToHDF5(csv_file,path):      
   h5_file = path / 'events.h5' # Save as a new "clean" file
   chunk_size = 500_000        

   # 1. Define your Column Names (Since you have no headers)
   column_names = ['x', 'y', 'polarity', 'timestamp']

   # 2. Define the "Trash" Pixel Coordinates
   # Replace these with your actual noisy pixel location
   BAD_X = 508   
   BAD_Y = 498

   # 3. Define Types
   schema_overrides = { 
      'x': pl.UInt16,
      'y': pl.UInt16,
      'polarity': pl.UInt8,
      'timestamp': pl.UInt64
   }

   # --- THE CONVERSION ---
   # print(f"Starting conversion with filtering: {csv_file} -> {h5_file}...")
   print(f"Removing all events at pixel ({BAD_X}, {BAD_Y})")

   if os.path.exists(h5_file):
      os.remove(h5_file)

   with h5py.File(h5_file, 'w') as f:
      dset = None
      row_count = 0
      dropped_count = 0
      
      reader = pl.read_csv_batched(
         csv_file, 
         batch_size=chunk_size, 
         has_header=False, 
         new_columns=column_names 
      )
      
      while True:
         batches = reader.next_batches(1)
         if not batches:
               break
         
         chunk = batches[0]
         initial_len = len(chunk)
         
         # --- CAST TYPES ---
         chunk = chunk.select([
               pl.col(name).cast(dtype) for name, dtype in schema_overrides.items()
         ])
         
         # --- THE FILTERING STEP ---
         # "Keep rows where X is NOT Bad OR Y is NOT Bad"
         # (This removes only rows where BOTH X and Y match the bad pixel)
         chunk = chunk.filter(
               ~((pl.col("x") == BAD_X) & (pl.col("y") == BAD_Y))
         )
         
         # Calculate how many we dropped (for your info)
         dropped_count += (initial_len - len(chunk))
         
         # If the chunk is empty after filtering (rare), skip it
         if len(chunk) == 0:
               continue
               
         data_numpy = chunk.to_numpy()
         
         # --- WRITE TO HDF5 ---
         if dset is None:
               dset = f.create_dataset(
                  'events', 
                  data=data_numpy, 
                  maxshape=(None, len(column_names)), 
                  chunks=True, 
                  compression="gzip"
               )
         else:
               dset.resize((dset.shape[0] + data_numpy.shape[0]), axis=0)
               dset[-data_numpy.shape[0]:] = data_numpy
               
         row_count += len(chunk)

         if dset is not None:
            dset.attrs['columns'] = column_names
            dset.attrs['hot_pixel_removed'] = [BAD_X, BAD_Y]

   print(f"Done! Total valid events: {row_count}. Total trash removed: {dropped_count}")
    
    
def runDecoder(path):
    print("start decodeing...")
    script_dir = Path(__file__).parent
    exepath = script_dir / "cnvt_raw_2_csv" / "build" / "Release" / "metavision_evt3_raw_file_decoder.exe"
    search_criteria = os.path.join(path,"*.raw")
    raw_files = glob.glob(search_criteria)
    for file_index in range(len(raw_files)):
      file = raw_files[file_index]
      output_path = script_dir.parent / "csv" / f"out{file_index}.csv"
      result = subprocess.run([exepath,file,output_path],capture_output=True,text=True)
      status = result.returncode

      if status == 0:
         print("command run successfully")
         convertToHDF5(output_path, path)
      else:
         print("command failed to run")

