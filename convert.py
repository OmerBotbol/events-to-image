from utils import decoder
from pathlib import Path

script_dir = Path(__file__).parent
events_dir = script_dir / "events"

for folder in sorted(events_dir.iterdir()):
    if not folder.is_dir():
        continue
    if (folder / "events.h5").exists():
        print(f"[skip]    {folder.name} — already has events.h5")
        continue
    print(f"[decode]  {folder.name}")
    decoder.runDecoder(folder)
