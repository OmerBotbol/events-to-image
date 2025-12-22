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

   # Actual column mapping
   column_names = ['x', 'y', 'polarity', 'timestamp']

   # Target types (Now we can actually enforce these separately!)
   types = {
      'x': pl.UInt16,
      'y': pl.UInt16,
      'polarity': pl.UInt8,
      'timestamp': pl.UInt64
}

   BAD_X, BAD_Y = 508, 498

   # print(f"Starting Columnar Conversion: {csv_file} -> {h5_file}...")

   if os.path.exists(h5_file):
      os.remove(h5_file)

   with h5py.File(h5_file, 'w') as f:
      # 1. Create the Group
      grp = f.create_group('events')
      
      # Dictionary to hold our 4 datasets
      dsets = {} 
      
      row_count = 0
      
      reader = pl.read_csv_batched(
         csv_file, batch_size=chunk_size, has_header=False, new_columns=column_names
      )
      
      while True:
         batches = reader.next_batches(1)
         if not batches: break
         chunk = batches[0]
         
         # Filter Hot Pixel
         chunk = chunk.filter(~((pl.col("x") == BAD_X) & (pl.col("y") == BAD_Y)))
         if len(chunk) == 0: continue

         # --- WRITE EACH COLUMN SEPARATELY ---
         for col_name in column_names:
               # Extract column and convert to numpy with CORRECT type
               target_type = types[col_name]
            
               # 2. Cast and Convert to Numpy
               data = chunk[col_name].cast(target_type).to_numpy()
               
               # Create Dataset on first run
               if col_name not in dsets:
                  dsets[col_name] = grp.create_dataset(
                     col_name,
                     data=data,
                     maxshape=(None,), # 1D array that can grow
                     chunks=True,
                     compression="gzip"
                  )
               # Resize and Append on subsequent runs
               else:
                  ds = dsets[col_name]
                  ds.resize((ds.shape[0] + data.shape[0]), axis=0)
                  ds[-data.shape[0]:] = data
         
         row_count += len(chunk)
         print(f"Processed {row_count} rows...")
         
      print("Conversion Complete.")
    
    
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

