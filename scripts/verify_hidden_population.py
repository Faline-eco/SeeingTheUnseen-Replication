"""Share of centrally invisible boxes that are genuinely hidden, over all flights."""
import json
from pathlib import Path
R = Path("/scratch/bambi/datasets/alfs_embed/zenodo_labels")
for mod in ("thermal", "rgb"):
    m = json.loads((R / f"{mod}_hidden_mask_all.json").read_text())
    tot = sum(len(v) for v in m.values())
    hid = sum(sum(1 for x in v if x) for v in m.values())
    print(f"{mod}: {tot} centrally invisible boxes, {hid} genuinely hidden "
          f"= {100*hid/tot:.1f} %  (not hidden: {tot-hid} = {100*(tot-hid)/tot:.1f} %)")
