"""Combine the detector's ball with the tracking's ball, in pitch metres.

The detector (vision/detect_fast.py, run over the whole match) gives up to 10 pixel candidates per
frame. The top one, when confident, is taken back to the pitch through the inverse homography.
A ball in the air lands beyond where it really is, the same way the tracking's own ground-plane
ball does, so neither is better on that count.

Modes, per frame:
    track   the tracking only (what my_fourth uses)
    fill    the tracking; the detector only where the tracking has no ball
    fix     fill, and also replace the tracking where a confident detection is more than
            `disagree` metres away from it (on a checked sample, the detector was right 6/6 times)
    det     the detector where confident; the tracking otherwise

Frames outside the detected spans fall back to the tracking, so this runs over the whole match
while only the detected spans change.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES  # noqa: E402
from load_homography import load as load_homography  # noqa: E402

DET_DIR = os.path.join(ROOT, "balldet", "detections_full")    # vision/detect_fast.py, whole match
MODES = ("track", "fill", "fix", "det")
# The scoreboard clock's colon looks like a ball to the detector; nothing in this box is the ball.
CLOCK_BOX = (175, 245, 28, 50)                                  # x0, x1, y0, y1 px


def detections(n, det_dir=DET_DIR):
    """(n, 3) best candidate per frame (x px, y px, conf) outside the clock; NaN where none."""
    top = np.full((n, 3), np.nan)
    for path in sorted(glob.glob(os.path.join(det_dir, "det_*.npz"))):
        data = np.load(path)
        start, dets = int(data["start"]), data["dets"].copy()
        x0, x1, y0, y1 = CLOCK_BOX
        clock = (dets[..., 0] > x0) & (dets[..., 0] < x1) & (dets[..., 1] > y0) & (dets[..., 1] < y1)
        dets[clock] = np.nan
        best = np.argmax(np.nan_to_num(dets[..., 2], nan=-1.0), axis=1)
        top[start:start + len(dets)] = dets[np.arange(len(dets)), best]
    return top


def to_pitch(px, py):
    """Pixels -> pitch metres through each frame's inverse homography. NaN in, NaN out."""
    matrices, _, valid = load_homography()
    inv = np.full_like(matrices, np.nan)
    inv[valid] = np.linalg.inv(matrices[valid])
    den = inv[:, 2, 0] * px + inv[:, 2, 1] * py + inv[:, 2, 2]
    x = (inv[:, 0, 0] * px + inv[:, 0, 1] * py + inv[:, 0, 2]) / den
    y = (inv[:, 1, 0] * px + inv[:, 1, 1] * py + inv[:, 1, 2]) / den
    length, width = GAMES[GAME]["pitch"]
    return np.stack([x * length, y * width], axis=1)


def fused_ball(track, mode="fix", conf=0.5, disagree=3.0):
    """(n, 2) ball in metres. `track` is the tracking's ball in metres, NaN where missing."""
    if mode not in MODES:
        raise ValueError("mode must be one of {}".format(MODES))
    if mode == "track":
        return track.copy()
    top = detections(len(track))
    det = to_pitch(top[:, 0], top[:, 1])
    confident = np.nan_to_num(top[:, 2]) >= conf
    has_track = ~np.isnan(track[:, 0])
    out = track.copy()
    if mode == "fill":
        use = confident & ~has_track
    elif mode == "fix":
        far = np.hypot(*(det - track).T) > disagree
        use = confident & (~has_track | np.nan_to_num(far, nan=True).astype(bool))
    else:
        use = confident
    out[use] = det[use]
    return out
