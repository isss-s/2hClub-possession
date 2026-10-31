"""Find what the round-1 detector gets wrong, to teach round 2.

    python vision/mine_hard.py            # ~3,000 frames, writes balldet/mined.npz

Round 1 was trained only on balls the classical snapper could find -- isolated, on open grass --
so it learned "small white blob on grass" and fires on white boots, pitch markings and specks.
This runs it on short runs of 3 consecutive frames spread over the match (never inside our
validation windows) and keeps two kinds of example:

  false hits   a confident candidate more than FAR_PX from a ball we know is right (a clean,
               consistent snap from vision/build_dataset.py). Round 2 sees these as "not a ball".
  hard balls   the top candidate on a frame where the snapper found nothing, within NEAR_PX of
               the tracking's ball, and steady across the 3 frames. These are the balls at feet
               and in crowds that round 1 never saw as positives.

Output: balldet/mined.npz with frame, kind (0 false hit, 1 hard ball), x, y, conf.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from build_dataset import consistent, excluded_frames, on_torso  # noqa: E402
from detect_fast import run_span  # noqa: E402
from games import require  # noqa: E402

N_RUNS = 1000
RUN = 3
FALSE_CONF = 0.2            # a hit at least this sure, far from the real ball, is a false hit
FAR_PX = 20
HARD_CONF = 0.3
NEAR_PX = 40                # the tracking misses a ground ball by 6-22 px
STEADY_PX = 6               # middle frame within this of the mean of its neighbours
SEED = 0


def main():
    from ultralytics import YOLO
    hints = np.load(os.path.join(ROOT, "balldet", "atd_ball_px.npz"))
    snapped = np.load(os.path.join(ROOT, "balldet", "snapped.npz"))["snapped"]
    feet = np.load(os.path.join(ROOT, "balldet", "players_px.npz"))["feet"]
    n = len(snapped)
    known = consistent(snapped) & ~on_torso(snapped, feet)

    # Graded play only: the two halves, as the public ground truth defines them.
    play = np.zeros(n, dtype=bool)
    play[11185:90652] = True
    play[110808:190198] = True
    allowed = play & ~excluded_frames(n)
    starts = np.flatnonzero(allowed[:-RUN] & allowed[RUN:])
    rng = np.random.default_rng(SEED)
    starts = np.sort(rng.choice(starts, N_RUNS, replace=False))

    model = YOLO(os.path.join(ROOT, "balldet", "best.pt"))
    video = require(["video"])[0]
    rows = []
    for count, start in enumerate(starts):
        dets = run_span(model, video, int(start), int(start) + RUN, "mps", 1280, False, RUN)
        for k in range(RUN):
            f = int(start) + k
            cands = dets[k][np.isfinite(dets[k][:, 2])]
            if known[f]:
                for x, y, conf in cands:
                    if conf >= FALSE_CONF and np.hypot(x - snapped[f, 0], y - snapped[f, 1]) > FAR_PX:
                        rows.append((f, 0, x, y, conf))
        tops = dets[:, 0]
        mid = int(start) + 1
        if (np.isnan(snapped[mid, 0]) and hints["seen"][mid] and np.all(tops[:, 2] >= HARD_CONF)
                and np.hypot(tops[1, 0] - hints["px"][mid], tops[1, 1] - hints["py"][mid]) < NEAR_PX
                and np.hypot(*(tops[1, :2] - (tops[0, :2] + tops[2, :2]) / 2)) < STEADY_PX):
            rows.append((mid, 1, tops[1, 0], tops[1, 1], tops[1, 2]))
        if count % 100 == 0:
            print("  run", count, "of", N_RUNS, "| examples so far", len(rows), flush=True)

    rows = np.array(rows, dtype=np.float64).reshape(-1, 5)
    np.savez_compressed(os.path.join(ROOT, "balldet", "mined.npz"), frame=rows[:, 0].astype(int),
                        kind=rows[:, 1].astype(int), x=rows[:, 2], y=rows[:, 3], conf=rows[:, 4])
    print("false hits", int((rows[:, 1] == 0).sum()), "| hard balls", int((rows[:, 1] == 1).sum()))


if __name__ == "__main__":
    main()
