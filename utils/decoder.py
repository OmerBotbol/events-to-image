import polars as pl
import numpy as np
from pathlib import Path
import h5py
import subprocess
import os
import glob

_BAD_X, _BAD_Y = 508, 498
_CHUNK = 2_000_000


def _make_wizard(Wizard, fpath):
    """
    Create an expelliarmus Wizard, trying both constructor signatures:
      - new API: Wizard(fpath=..., encoding="evt3")
      - old API: Wizard(encoding="evt3")  then pass fpath to read()
    Returns (wizard, fpath_for_read) where fpath_for_read is None if the
    constructor already consumed it.
    """
    try:
        wizard = Wizard(fpath=str(fpath), encoding="evt3")
        return wizard, None          # fpath was consumed by constructor
    except TypeError:
        wizard = Wizard(encoding="evt3")
        return wizard, str(fpath)    # fpath must be passed to read()


def _iter_chunks(wizard, fpath_for_read, chunk_size):
    """
    Yield event chunks handling every known expelliarmus read() signature:
      n_events=N  |  chunk_size=N  |  no size arg (read all, slice manually)
    fpath_for_read is None when the path was already given to the constructor.
    """
    import inspect
    params = set(inspect.signature(wizard.read).parameters.keys())

    # Build positional args and keyword args
    pos  = [fpath_for_read] if fpath_for_read is not None else []
    kwds = {}
    if "n_events"   in params: kwds["n_events"]   = chunk_size
    elif "chunk_size" in params: kwds["chunk_size"] = chunk_size

    try:
        result = wizard.read(*pos, **kwds)
    except TypeError as exc:
        raise RuntimeError(
            f"Could not call expelliarmus Wizard.read() — unsupported API.\n"
            f"  Tried: read({', '.join(repr(p) for p in pos)}, {kwds})\n"
            f"  Error: {exc}\n"
            f"  Try: pip install --upgrade expelliarmus"
        ) from exc

    # result may be a generator (chunked) or a plain array (all-at-once)
    if hasattr(result, "__next__") or hasattr(result, "__iter__") and not isinstance(result, np.ndarray):
        yield from result
    else:
        # Plain array — slice manually
        for i in range(0, max(1, len(result)), chunk_size):
            yield result[i : i + chunk_size]


def _extract_fields(chunk):
    """
    Extract (t, x, y, p) from a chunk, tolerating different field naming
    conventions used across expelliarmus versions.
    """
    names = chunk.dtype.names
    def pick(candidates):
        for c in candidates:
            if c in names:
                return chunk[c]
        raise KeyError(f"None of {candidates} found in chunk fields {names}")

    t = pick(("t", "timestamp", "ts"))
    x = pick(("x", "x_addr"))
    y = pick(("y", "y_addr"))
    p = pick(("p", "pol", "polarity"))
    return t, x, y, p

def convertToHDF5(csv_file, path):
   h5_file = path / 'events.h5'
   # Larger batches = fewer resize calls and less HDF5 bookkeeping overhead
   batch_size = 2_000_000

   column_names = ['x', 'y', 'polarity', 'timestamp']
   types = {
      'x':         pl.UInt16,
      'y':         pl.UInt16,
      'polarity':  pl.Int8,
      'timestamp': pl.Float64,
   }
   BAD_X, BAD_Y = 508, 498

   if os.path.exists(h5_file):
      os.remove(h5_file)

   with h5py.File(h5_file, 'w') as f:
      grp  = f.create_group('events')
      dsets = {}
      row_count = 0

      reader = pl.read_csv_batched(
         csv_file, batch_size=batch_size, has_header=False, new_columns=column_names
      )

      while True:
         batches = reader.next_batches(1)
         if not batches:
            break
         chunk = batches[0]

         chunk = chunk.filter(~((pl.col("x") == BAD_X) & (pl.col("y") == BAD_Y)))
         if len(chunk) == 0:
            continue

         chunk = chunk.with_columns([
            (pl.col("timestamp") / 1_000_000.0).alias("timestamp"),
            (pl.col("polarity").cast(pl.Int8) * 2 - 1).alias("polarity"),
         ])

         for col_name in column_names:
            data = chunk[col_name].cast(types[col_name]).to_numpy()

            if col_name not in dsets:
               # chunk size aligned to batch size; gzip level 1 is ~4x faster
               # than the default (level 4) with negligible difference in file size
               dsets[col_name] = grp.create_dataset(
                  col_name,
                  data=data,
                  maxshape=(None,),
                  chunks=(batch_size,),
                  compression="gzip",
                  compression_opts=1,
               )
            else:
               ds = dsets[col_name]
               ds.resize((ds.shape[0] + data.shape[0]), axis=0)
               ds[-data.shape[0]:] = data

         row_count += len(chunk)
         print(f"Processed {row_count:,} rows...")

      print("Conversion complete.")
    
    
def runDecoderFast(path, use_gpu=False):
    """
    Decode .raw → events.h5 directly, skipping the CSV intermediate entirely.

    Why this is faster
    ------------------
    The standard pipeline writes every event as a text row to a CSV file
    (≈20 bytes/event × 1 B events = ~20 GB), then reads and parses that file.
    This function decodes the binary EVT3 format directly into HDF5 using
    `expelliarmus`, cutting out the huge intermediate I/O step.

    Requirements
    ------------
    pip install expelliarmus          # EVT3 binary reader
    pip install cupy-cuda12x          # GPU array ops (optional; match your CUDA version)

    GPU note
    --------
    EVT3 timestamps are delta-encoded (each timestamp = previous + delta), so
    the binary parsing itself is sequential and cannot be GPU-parallelised.
    GPU acceleration applies to the per-chunk array operations (filtering,
    type conversion) which gives a modest extra speedup on top of the main
    CSV-skip benefit.
    """
    try:
        from expelliarmus import Wizard
    except ImportError:
        raise ImportError(
            "expelliarmus is not installed.\n"
            "Install it with:  pip install expelliarmus\n"
            "Then retry, or uncheck 'Fast decode' to use the standard decoder."
        )

    xp = np
    if use_gpu:
        try:
            import cupy as cp
            xp = cp
            print("GPU mode active — using CuPy for array operations.")
        except ImportError:
            print("CuPy not found; falling back to CPU.\n"
                  "Install with:  pip install cupy-cuda12x  (match your CUDA version)")

    h5_file = path / "events.h5"
    if os.path.exists(h5_file):
        os.remove(h5_file)

    raw_files = glob.glob(os.path.join(path, "*.raw"))
    if not raw_files:
        print("No .raw files found.")
        return

    for raw_file in raw_files:
        print(f"Decoding  {raw_file}  (direct RAW → HDF5) …")
        wizard, fpath_for_read = _make_wizard(Wizard, raw_file)

        with h5py.File(h5_file, "w") as f:
            grp   = f.create_group("events")
            dsets = {}
            total = 0

            for chunk in _iter_chunks(wizard, fpath_for_read, _CHUNK):
                t_raw, x, y, p = _extract_fields(chunk)

                if use_gpu:
                    import cupy as cp
                    t_raw = cp.asarray(t_raw);  x = cp.asarray(x)
                    y     = cp.asarray(y);      p = cp.asarray(p)

                # Filter bad pixels
                mask = ~((x == _BAD_X) & (y == _BAD_Y))
                t_raw, x, y, p = t_raw[mask], x[mask], y[mask], p[mask]
                if len(t_raw) == 0:
                    continue

                # timestamp μs (int64) → seconds (float64);  polarity 0/1 → ±1
                t_s  = t_raw.astype(xp.float64) / 1_000_000.0
                p_pm = p.astype(xp.int8) * 2 - 1

                if use_gpu:
                    import cupy as cp
                    t_s  = cp.asnumpy(t_s);  x = cp.asnumpy(x)
                    y    = cp.asnumpy(y);    p_pm = cp.asnumpy(p_pm)

                arrays = {
                    "timestamp": (t_s,  np.float64),
                    "x":         (x,    np.uint16),
                    "y":         (y,    np.uint16),
                    "polarity":  (p_pm, np.int8),
                }
                for name, (arr, dtype) in arrays.items():
                    arr = arr.astype(dtype)
                    if name not in dsets:
                        dsets[name] = grp.create_dataset(
                            name, data=arr, maxshape=(None,),
                            chunks=(_CHUNK,), compression="gzip", compression_opts=1,
                        )
                    else:
                        ds = dsets[name]
                        ds.resize((ds.shape[0] + len(arr)), axis=0)
                        ds[-len(arr):] = arr

                total += len(t_s)
                print(f"  {total:,} events …", end="\r")

        if total == 0:
            os.remove(h5_file)
            raise RuntimeError(
                "Fast decoder produced 0 events. "
                "Check that the .raw file is a valid EVT3 recording."
            )
        print(f"\nDone — {total:,} events written to {h5_file}")


def runDecoder(path):
    print("start decodeing...")
    script_dir = Path(__file__).parent
    exepath = script_dir / "cnvt_raw_2_csv" / "build" / "Release" / "metavision_evt3_raw_file_decoder.exe"
    search_criteria = os.path.join(path,"*.raw")
    raw_files = glob.glob(search_criteria)
    for file_index in range(len(raw_files)):
      file = raw_files[file_index]
      output_path = script_dir.parent / "csv" / f"out{file_index}.csv"
      result = subprocess.run([exepath, file, output_path],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      status = result.returncode

      if status == 0:
         print("command run successfully")
         convertToHDF5(output_path, path)
      else:
         print("command failed to run")

