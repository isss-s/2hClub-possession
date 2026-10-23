"""My third version: possession from ball *control*, not just the nearest player.

On top of the tracking:
  1. Control: a team is on the ball when one of its players is within --radius metres of it.
     Possession only changes after the other team has been on the ball for --switch frames in a
     row. When nobody is on the ball (a pass in flight, a loose ball), the last team keeps it.
  2. Ball out of play: the ball seen outside the pitch lines is a dead ball.
  3. Ball missing: a gap longer than --gap frames is a dead ball. If --gap-long is set, a gap is
     only dead once it is longer than that.
  4. Slow ball: a median ball speed under --slow m/s for at least --min-dead frames is a dead ball.
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
from baseline import team_of_each_track, write_submission  # noqa: E402
from games import GAME, GAMES, require, source_frames  # noqa: E402
from my_second import drop_short  # noqa: E402

HOME, AWAY, NOBODY, LOOSE = 0, 1, -1, -2


def read_tracking():
    """Ball (n, 2) and players (n, slots, 2) in metres. NaN where not tracked."""
    length, width = GAMES[GAME]["pitch"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    n = source_frames("tactical")
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, len(team), 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            frame, groups, ball_field = line.split(":")
            i = int(frame) - 1
            for slot, group in enumerate(groups.split(";")):
                if group[0] != "N":
                    values = group.split(",")
                    players[i, slot, 0] = float(values[0]) * length
                    players[i, slot, 1] = float(values[1]) * width
            if ball_field[0] != "N":
                values = ball_field.split(",")
                ball[i, 0] = float(values[0]) * length
                ball[i, 1] = float(values[1]) * width
    return ball, players, team


def runs(mask):
    """(start, end) of every run of True, end exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def control(ball, players, team, radius):
    """Per frame: HOME/AWAY if that team has a player within `radius` m, NOBODY if no ball."""
    dist = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))
    dist = np.where(np.isnan(dist), np.inf, dist)
    home = np.where(team[None, :] == HOME, dist, np.inf).min(1)
    away = np.where(team[None, :] == AWAY, dist, np.inf).min(1)
    out = np.full(len(ball), LOOSE, dtype=np.int8)
    close_home, close_away = home < radius, away < radius
    out[close_home & ~close_away] = HOME
    out[close_away & ~close_home] = AWAY
    both = close_home & close_away
    out[both] = np.where(home[both] <= away[both], HOME, AWAY)
    out[np.isnan(ball[:, 0])] = NOBODY
    return out


def dead_frames(ball, gap, gap_long, out_margin):
    """Ball outside the lines, or missing for too long."""
    length, width = GAMES[GAME]["pitch"]
    seen = ~np.isnan(ball[:, 0])
    with np.errstate(invalid="ignore"):
        dead = seen & ((ball[:, 0] < -out_margin) | (ball[:, 0] > length + out_margin) |
                       (ball[:, 1] < -out_margin) | (ball[:, 1] > width + out_margin))
    for start, end in runs(~seen):
        if start == 0 or end >= len(ball):
            continue
        if end - start > gap and (gap_long is None or end - start > gap_long):
            dead[start:end] = True
    return dead


def walk(ctrl, dead, switch):
    """Carry the possessing team forward; switch after `switch` frames of the other team."""
    out = np.full(len(ctrl), NOBODY, dtype=np.int8)
    current, run, candidate = NOBODY, 0, NOBODY
    for i in range(len(ctrl)):
        if dead[i]:
            current, run = NOBODY, 0
            continue
        c = ctrl[i]
        if c >= 0:
            if current == NOBODY:
                current, run = c, 0
            elif c != current:
                if c == candidate:
                    run += 1
                else:
                    candidate, run = c, 1
                if run >= switch:
                    current = c
                    out[i - run + 1:i] = current  # the change started back there
                    run = 0
            else:
                run = 0
        out[i] = current
    return out


def slow_ball(ball, window, threshold, min_len):
    speed = np.full(len(ball), np.nan)
    speed[1:] = np.hypot(np.diff(ball[:, 0]), np.diff(ball[:, 1])) * 25
    median = pd.Series(speed).rolling(2 * window + 1, center=True, min_periods=window).median()
    return drop_short((median < threshold).values, min_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=1.5, help="metres; 'on the ball'")
    parser.add_argument("--switch", type=int, default=25, help="frames; 25 = 1 second")
    parser.add_argument("--gap", type=int, default=50, help="frames")
    parser.add_argument("--gap-long", type=int, default=150, help="frames")
    parser.add_argument("--out-margin", type=float, default=0.5, help="metres past the line")
    parser.add_argument("--window", type=int, default=125, help="frames each side for the median")
    parser.add_argument("--slow", type=float, default=3.0, help="m/s")
    parser.add_argument("--min-dead", type=int, default=300, help="frames")
    args = parser.parse_args()

    print("reading tracking ...")
    ball, players, team = read_tracking()
    ctrl = control(ball, players, team, args.radius)
    dead = dead_frames(ball, args.gap, args.gap_long, args.out_margin)
    codes = walk(ctrl, dead, args.switch)
    labels = np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    labels[slow_ball(ball, args.window, args.slow, args.min_dead)] = "D"

    path = write_submission(os.path.join(ROOT, "evaluation", "results", "my_third.csv"), labels)
    print("wrote", os.path.relpath(path, ROOT))
