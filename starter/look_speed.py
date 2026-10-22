"""How fast is the ball during each stretch of the public 5 minutes?

Speed is worked out from the ball's position, frame to frame, in metres per second.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

length, width = GAMES[GAME]["pitch"]
raw_path = require(["atd_raw"])[0]
n = source_frames("tactical")
x = np.full(n, np.nan)
y = np.full(n, np.nan)
example = None
with open(raw_path) as handle:
    for line in handle:
        parts = line.split(":")
        if len(parts) == 3 and parts[2][0] != "N":
            values = parts[2].split(",")
            x[int(parts[0]) - 1] = float(values[0]) * length
            y[int(parts[0]) - 1] = float(values[1]) * width
            example = example or parts[2].strip()

print("example ball field from the file: {}\n".format(example))

# Distance moved between consecutive frames x 25 frames per second = metres per second.
# NaN wherever the ball is missing in either frame.
speed = np.full(n, np.nan)
speed[1:] = np.hypot(np.diff(x), np.diff(y)) * 25

gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
labels = gt["label"].values
print("label  frames         secs  ball seen  median speed  slow (<1 m/s)")
start = 0
for frame in range(1, len(labels) + 1):
    if frame == len(labels) or labels[frame] != labels[start]:
        if labels[start] != "X":
            s = speed[start:frame]
            seen = s[~np.isnan(s)]
            print("  {}    {:>6}-{:<6} {:>5.1f}     {:>4.0f}%        {:>5}       {:>4.0f}%".format(
                labels[start], start, frame - 1, (frame - start) / 25.0,
                100.0 * len(seen) / len(s),
                "{:.1f}".format(np.median(seen)) if len(seen) else "-",
                100.0 * np.mean(seen < 1.0) if len(seen) else 0))
        start = frame
