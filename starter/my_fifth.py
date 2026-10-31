"""my_fourth's rules on a better ball: the tracking's ball combined with our ball detector.

    python starter/my_fifth.py --mode fix                # writes evaluation/results/my_fifth_fix.csv

Only "who is on the ball" uses the combined ball. The dead-ball rules keep the tracking's own
ball, on purpose: the tracking tends to drop the ball during stoppages even when it is lying in
plain sight, and my_fourth's "ball missing while the players walk" rule depends on exactly that.
Filling those gaps with the detector erased the stoppages (w2 fell from 0.77 to 0.24). Everything
else is my_fourth unchanged, so any change in score comes from the ball alone. See vision/fuse.py
for the modes. Needs balldet/detections_full/ from vision/detect_fast.py.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vision"))
from baseline import write_submission  # noqa: E402
from fuse import MODES, fused_ball  # noqa: E402
from my_fourth import dead_frames, players_speed  # noqa: E402
from my_third import AWAY, HOME, control, read_tracking, slow_ball, walk  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="fix")
    parser.add_argument("--conf", type=float, default=0.5, help="detector confidence to trust")
    parser.add_argument("--disagree", type=float, default=3.0,
                        help="metres; 'fix' replaces the tracking ball beyond this")
    parser.add_argument("--radius", type=float, default=1.5, help="metres; 'on the ball'")
    parser.add_argument("--switch", type=int, default=25, help="frames; 25 = 1 second")
    parser.add_argument("--gap", type=int, default=50, help="frames")
    parser.add_argument("--players-slow", type=float, default=1.3, help="m/s")
    parser.add_argument("--players-window", type=int, default=125, help="frames")
    parser.add_argument("--out-margin", type=float, default=0.5, help="metres past the line")
    parser.add_argument("--window", type=int, default=125, help="frames each side for the median")
    parser.add_argument("--slow", type=float, default=3.0, help="m/s")
    parser.add_argument("--min-dead", type=int, default=300, help="frames")
    parser.add_argument("--name", default=None, help="output name; default my_fifth_<mode>")
    args = parser.parse_args()

    track, players, team = read_tracking()
    ball = fused_ball(track, args.mode, args.conf, args.disagree)
    ctrl = control(ball, players, team, args.radius)
    team_speed = players_speed(players, args.players_window)
    dead = dead_frames(track, team_speed, args.gap, args.players_slow, args.out_margin)
    codes = walk(ctrl, dead, args.switch)
    labels = np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    labels[slow_ball(track, args.window, args.slow, args.min_dead)] = "D"

    name = args.name or "my_fifth_" + args.mode
    path = write_submission(os.path.join(ROOT, "evaluation", "results", name + ".csv"), labels)
    print("wrote", os.path.relpath(path, ROOT))
