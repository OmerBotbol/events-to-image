import pandas as pd
from pathlib import Path
import scipy.io
import subprocess
import os
import glob

def readLargeFile(csv_file,path):
   chunk_size = 10_000_000
   reader = pd.read_csv(csv_file, chunksize=chunk_size, na_values=['NA'])
   buffer_list = []
   current_buffer_size = 0
   counter = 0

   print("Starting to process...")

    # Iterate through the file in chunks
   for data_chunk in reader:
        
        # --- Logic: Filter the data ---
        
        # Note: .iloc[:, 0] means all rows, 0th column (MATLAB's column 1)
       condition = (data_chunk.iloc[:, 0] == 508) & (data_chunk.iloc[:, 1] == 498)
        
        # Keep only rows where the condition is False (remove the matches)
       data_chunk_filtered = data_chunk[~condition]
        
        # Add to our buffer
       buffer_list.append(data_chunk_filtered)
       current_buffer_size += len(data_chunk_filtered)
        
        # --- Logic: Check size and Save ---
       if current_buffer_size > chunk_size:
            # Combine the list of chunks into one DataFrame
           full_chunk = pd.concat(buffer_list)
            
            # Convert to numpy array (to match table2array behavior for the .mat file)
           data_to_save = full_chunk.to_numpy()
            
            # Save to .mat file
           filename = os.path.join(path, f'dataChunk{counter}.mat')
           scipy.io.savemat(filename, {'dataChunk': data_to_save})
           print(f"Saved: {filename}")
            
            # Reset counters and buffer
           counter += 1
           buffer_list = []
           current_buffer_size = 0

    # --- Save any remaining data after the loop finishes ---
   if buffer_list:
       full_chunk = pd.concat(buffer_list)
       data_to_save = full_chunk.to_numpy()
        
       filename = os.path.join(path, f'dataChunk{counter}.mat')
       scipy.io.savemat(filename, {'dataChunk': data_to_save})
       print(f"Saved: {filename}")

   print("Saved")
    
    
def runDecoder(path):
    script_dir = Path(__file__).parent
    exepath = script_dir / "cnvt_raw_2_csv" / "build" / "Release" / "metavision_evt3_raw_file_decoder.exe"
    chunk_path = path / "chunks"
    search_criteria = os.path.join(path,"*.raw")
    raw_files = glob.glob(search_criteria)
    for file_index in range(len(raw_files)):
      file = raw_files[file_index]
      output_path = script_dir.parent / "csv" / f"out{file_index}.csv"
      result = subprocess.run([exepath,file,output_path],capture_output=True,text=True)
      status = result.returncode

      if status == 0:
         print("command run successfully")
         readLargeFile(output_path, chunk_path)
      else:
         print("command failed to run")

