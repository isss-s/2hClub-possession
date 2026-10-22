"""My second version: my_first plus a slow-ball dead-ball rule.

When the ball's typical speed over the surrounding seconds is low, play is probably stopped
(the ball is waiting for a goal kick or free kick), so those frames become D.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from baseline import AWAY, HOME, NOBODY, build, write_submission  # noqa: E402
from games import GAME, GAMES, require, source_frames  # noqa: E402
from my_first import fill_gaps, hysteresis  # noqa: E402


def ball_speed():
    """Ball speed in m/s for every frame, from its position frame to frame. NaN when missing."""
    length, width = GAMES[GAME]["pitch"]
    n = source_frames("tactical")
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    with open(require(["atd_raw"])[0]) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) == 3 and parts[2][0] != "N":
                values = parts[2].split(",")
                x[int(parts[0]) - 1] = float(values[0]) * length
                y[int(parts[0]) - 1] = float(values[1]) * width
    speed = np.full(n, np.nan)
    speed[1:] = np.hypot(np.diff(x), np.diff(y)) * 25
    return speed


def slow_stretches(speed, window, threshold):
    """True where the median ball speed over +/- `window` frames is below `threshold` m/s."""
    median = pd.Series(speed).rolling(2 * window + 1, center=True, min_periods=window).median()
    return (median < threshold).values


def drop_short(mask, min_len):
    """Keep only the True runs that last at least `min_len` frames."""
    out = mask.copy()
    n, i = len(out), 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i
        while j < n and out[j]:
            j += 1
        if j - i < min_len:
            out[i:j] = False
        i = j
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", type=int, default=50, help="frames; 25 = 1 second")
    parser.add_argument("--switch", type=int, default=50, help="frames; 25 = 1 second")
    parser.add_argument("--window", type=int, default=125, help="frames each side for the median")
    parser.add_argument("--slow", type=float, default=3.0, help="m/s; below this counts as dead")
    parser.add_argument("--min-dead", type=int, default=0,
                        help="frames; slow stretches shorter than this are ignored")
    args = parser.parse_args()

    codes = build()
    codes = fill_gaps(codes, args.gap)
    codes = hysteresis(codes, args.switch)
    labels = np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))

    print("working out ball speed ...")
    dead = slow_stretches(ball_speed(), args.window, args.slow)
    dead = drop_short(dead, args.min_dead)
    labels[dead] = "D"
    print("  {} frames turned into D by the slow-ball rule".format(int(dead.sum())))

    name = "my_second_w{}_slow{}_min{}".format(args.window, args.slow, args.min_dead)
    path = write_submission(os.path.join(ROOT, "evaluation", "results", name + ".csv"), labels)
    print("wrote", os.path.relpath(path, ROOT))
