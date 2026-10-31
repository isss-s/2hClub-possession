"""Where the provided tracking puts the ball, in pixels of the tactical video.

    python vision/ball_track.py          # writes balldet/atd_ball_px.npz (gitignored, local only)

The ATD ball and the homography share the same normalised 0-1 pitch coordinates, so projecting
one through the other gives the ball's pixel on every frame the tracking saw it. These are the
free labels the ball detector is trained on.

The tracking reports the ball on the ground plane. A ball in the air projects below where it
really is, which is one reason vision/build_dataset.py keeps only frames where the ball moves
slowly enough to be on the ground.

Never commit the output: per-frame ball positions are derived tracking data (RULES.md).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from load_homography import load as load_homography  # noqa: E402

OUT_DIR = os.path.join(ROOT, "balldet")
OUT_PATH = os.path.join(OUT_DIR, "atd_ball_px.npz")
PLAYERS_PATH = os.path.join(OUT_DIR, "players_px.npz")
BALL_DIAMETER_M = 0.22


def read_tracking():
    """Normalised ball (n, 3: x, y, speed) and players (n, slots, 2). NaN where not tracked."""
    raw_path = require(["atd_raw"])[0]
    n = source_frames("tactical")
    ball = np.full((n, 3), np.nan, dtype=np.float64)
    players = None
    with open(raw_path) as handle:
        for line in handle:
            frame, groups, ball_field = line.split(":")
            i = int(frame) - 1
            groups = groups.split(";")
            if players is None:
                players = np.full((n, len(groups), 2), np.nan, dtype=np.float64)
            for slot, group in enumerate(groups):
                if group[0] != "N":
                    players[i, slot] = [float(v) for v in group.split(",")[:2]]
            if ball_field[0] != "N":
                ball[i] = [float(v) for v in ball_field.split(",")[:3]]
    return ball, players


def project_all(matrices, x, y):
    """Vectorised load_homography.project over frames: (n,3,3), (n,), (n,) -> two (n,) arrays."""
    den = matrices[:, 2, 0] * x + matrices[:, 2, 1] * y + matrices[:, 2, 2]
    return ((matrices[:, 0, 0] * x + matrices[:, 0, 1] * y + matrices[:, 0, 2]) / den,
            (matrices[:, 1, 0] * x + matrices[:, 1, 1] * y + matrices[:, 1, 2]) / den)


def main():
    length, width = GAMES[GAME]["pitch"]
    ball, players = read_tracking()
    matrices, error, valid = load_homography()
    # Players project to their feet (the tracking is on the ground plane). build_dataset.py uses
    # them to throw out "balls" that are really shirt numbers or socks above someone's feet.
    feet = np.full(players.shape, np.nan, dtype=np.float32)
    for slot in range(players.shape[1]):
        feet[:, slot, 0], feet[:, slot, 1] = project_all(matrices, players[:, slot, 0],
                                                         players[:, slot, 1])
    px, py = project_all(matrices, ball[:, 0], ball[:, 1])
    # Ball size in pixels: the image distance covered by one ball diameter along the pitch length
    # at that spot. Good enough to size a box; it ignores height and perspective across the ball.
    ex, ey = project_all(matrices, ball[:, 0] + BALL_DIAMETER_M / length, ball[:, 1])
    diameter = np.hypot(ex - px, ey - py)
    seen = ~np.isnan(px) & valid
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(OUT_PATH, px=px, py=py, diameter=diameter, seen=seen,
                        pitch_x=ball[:, 0] * length, pitch_y=ball[:, 1] * width,
                        speed=ball[:, 2], reprojection_error=error)
    np.savez_compressed(PLAYERS_PATH, feet=feet)
    print("ball tracked on {} of {} frames; median diameter {:.1f} px; wrote {}".format(
        int(seen.sum()), len(seen), float(np.nanmedian(diameter[seen])),
        os.path.relpath(OUT_PATH, ROOT)))


if __name__ == "__main__":
    main()
