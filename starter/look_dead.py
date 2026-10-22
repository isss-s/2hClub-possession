"""Look at the dead-ball stretches in the public 5 minutes, next to the Smart Tagging."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load_smart_tagging import load  # noqa: E402

ROOT = os.path.dirname(HERE)
gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
mine = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "my_first_gap50_switch50.csv"))
events = load()

labels = gt["label"].values
start = None
for frame in range(len(labels) + 1):
    is_dead = frame < len(labels) and labels[frame] == "D"
    if is_dead and start is None:
        start = frame
    if not is_dead and start is not None:
        end = frame  # first frame after the dead stretch
        mine_here = mine["label"].values[start:end]
        print("\nDEAD BALL  frames {}-{}  ({:.1f} s)".format(start, end - 1, (end - start) / 25))
        print("  my labels here:  H {}  A {}  D {}".format(
            (mine_here == "H").sum(), (mine_here == "A").sum(), (mine_here == "D").sum()))
        print("  tags within 10 s:")
        for e in events:
            if e["end_frame"] >= start - 250 and e["start_frame"] <= end + 250:
                print("    {:<22} {:<24} team {:<14} frames {}-{}".format(
                    e["code"], e["tags"].get("Type") or "", e["tags"].get("Team") or "",
                    e["start_frame"], e["end_frame"]))
        start = None
