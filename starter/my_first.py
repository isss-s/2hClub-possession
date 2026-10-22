"""My first improvement on the baseline.

Same nearest-player idea, two changes:
  1. A short ball-missing gap is not a dead ball: the ball is usually just hidden.
     Gaps up to --gap frames keep the team that had it before.
  2. Possession only changes team after the other team has been nearest for
     --switch frames in a row, so one touch or a passing opponent does not flip it.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baseline import AWAY, HOME, NOBODY, ROOT, build, write_submission  # noqa: E402


def fill_gaps(codes, gap):
    """Fill runs of NOBODY no longer than `gap` with the team before them."""
    out = codes.copy()
    n, i = len(out), 0
    while i < n:
        if out[i] != NOBODY:
            i += 1
            continue
        j = i
        while j < n and out[j] == NOBODY:
            j += 1
        if i > 0 and j < n and j - i <= gap:
            out[i:j] = out[i - 1]
        i = j
    return out


def hysteresis(codes, switch):
    """Change team only after `switch` frames in a row of the other team."""
    out = codes.copy()
    current, run = NOBODY, 0
    for k in range(len(codes)):
        c = codes[k]
        if c == NOBODY:
            current, run = NOBODY, 0
            continue
        if current == NOBODY or c == current:
            current, run = c, 0
        else:
            run += 1
            if run >= switch:
                current, run = c, 0
                out[k - switch + 1:k + 1] = current  # the change started back there
                continue
        out[k] = current
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", type=int, default=50, help="frames; 25 = 1 second")
    parser.add_argument("--switch", type=int, default=25, help="frames; 25 = 1 second")
    args = parser.parse_args()

    codes = build()
    codes = fill_gaps(codes, args.gap)
    codes = hysteresis(codes, args.switch)
    labels = np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))

    name = "my_first_gap{}_switch{}".format(args.gap, args.switch)
    path = write_submission(os.path.join(ROOT, "evaluation", "results", name + ".csv"), labels)
    print("wrote", os.path.relpath(path, ROOT))
    print("Now score it:  python evaluation/run_evaluation.py")
