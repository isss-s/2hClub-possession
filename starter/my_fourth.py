"""My fourth version: my_third, but a long ball gap is only dead if the players slow down.

When the tracking loses the ball for more than --gap frames, my_third called it a dead ball. In
some stretches the ball is missing for 20 s or more while play goes on. The players tell the two
apart: in live play they run (median ~2.2 m/s), at a stoppage they walk (~1.0 m/s). So inside a
long gap, a frame is dead only where the players' typical speed is below --players-slow m/s.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from baseline import write_submission  # noqa: E402
from games import GAME, GAMES  # noqa: E402
from my_third import (AWAY, HOME, control, read_tracking, runs, slow_ball,  # noqa: E402
                      walk)

MAX_PLAYER_SPEED = 12.0  # m/s; faster than any player, so a tracking jump


def players_speed(players, window):
    """Median over +/- window/2 frames of the mean speed of all tracked players, in m/s."""
    step = np.sqrt((np.diff(players, axis=0) ** 2).sum(-1)) * 25
    step = np.vstack([np.full((1, players.shape[1]), np.nan), step])
    step[step > MAX_PLAYER_SPEED] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # frames with no players
        mean = np.nanmean(step, axis=1)
    return pd.Series(mean).rolling(window, center=True, min_periods=window // 5).median().values


def dead_frames(ball, team_speed, gap, players_slow, out_margin):
    """Ball outside the lines, or missing for longer than `gap` while the players are slow."""
    length, width = GAMES[GAME]["pitch"]
    seen = ~np.isnan(ball[:, 0])
    with np.errstate(invalid="ignore"):
        dead = seen & ((ball[:, 0] < -out_margin) | (ball[:, 0] > length + out_margin) |
                       (ball[:, 1] < -out_margin) | (ball[:, 1] > width + out_margin))
        for start, end in runs(~seen):
            if start == 0 or end >= len(ball) or end - start <= gap:
                continue
            dead[start:end] = team_speed[start:end] < players_slow
    return dead


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=1.5, help="metres; 'on the ball'")
    parser.add_argument("--switch", type=int, default=25, help="frames; 25 = 1 second")
    parser.add_argument("--gap", type=int, default=50, help="frames")
    parser.add_argument("--players-slow", type=float, default=1.3, help="m/s")
    parser.add_argument("--players-window", type=int, default=125, help="frames")
    parser.add_argument("--out-margin", type=float, default=0.5, help="metres past the line")
    parser.add_argument("--window", type=int, default=125, help="frames each side for the median")
    parser.add_argument("--slow", type=float, default=3.0, help="m/s")
    parser.add_argument("--min-dead", type=int, default=300, help="frames")
    args = parser.parse_args()

    print("reading tracking ...")
    ball, players, team = read_tracking()
    ctrl = control(ball, players, team, args.radius)
    team_speed = players_speed(players, args.players_window)
    dead = dead_frames(ball, team_speed, args.gap, args.players_slow, args.out_margin)
    codes = walk(ctrl, dead, args.switch)
    labels = np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    labels[slow_ball(ball, args.window, args.slow, args.min_dead)] = "D"

    path = write_submission(os.path.join(ROOT, "evaluation", "results", "my_fourth.csv"), labels)
    print("wrote", os.path.relpath(path, ROOT))
