"""Build the round-1 ball-detector training set from the tracking, with no hand labelling.

    python vision/ball_track.py        # once: tracking ball -> pixel hints
    python vision/build_dataset.py     # two passes over the video, a few minutes each

Pass 1 snaps every tracked ball to the real ball in the image (vision/snap.py) and saves the
result to balldet/snapped.npz. A label is kept only when the snap is unambiguous and agrees with
the snaps on the frames either side (straight-line motion within 4 px), which on a hand-checked
sample of 24 was right 24 times. Snaps sitting over a player's torso
(shirt numbers, socks) are dropped too, using the tracked players projected to their feet.

Pass 2 cuts 640x640 tiles at full resolution -- the ball is ~6 px wide, so the frame is never
downscaled -- with the ball at a random spot, plus ball-free tiles from the same frames as
negatives. Frames inside our validation windows (and the public window) are left out entirely so
the possession scores stay honest, and whole 5-minute blocks are held out as the detector's own
validation split.

Output (gitignored, local only -- tiles are clipped video, which RULES.md forbids publishing):
    balldet/yolo/{train,val}/{images,labels}, balldet/yolo/data.yaml
"""
import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import require  # noqa: E402
from score_windows import LABELS_DIR, read_marks  # noqa: E402
from snap import snap  # noqa: E402

OUT = os.path.join(ROOT, "balldet")
YOLO_DIR = os.path.join(OUT, "yolo")
TILE = 640
WIDTH, HEIGHT = 1920, 1080
PUBLIC_WINDOW = (11185, 18685)
WINDOW_MARGIN = 250          # frames kept away from each validation window
VAL_BLOCK = 7500             # detector validation: every 5th block of 5 minutes
CONSISTENT_PX = 4.0
TORSO_DX, TORSO_DY = 12, (12, 60)


def excluded_frames(n):
    """True on frames inside (or near) any possession-validation window."""
    out = np.zeros(n, dtype=bool)
    spans = [PUBLIC_WINDOW]
    for path in glob.glob(os.path.join(LABELS_DIR, "*.csv")):
        start, end, _ = read_marks(path)
        spans.append((start, end))
    for start, end in spans:
        out[max(0, start - WINDOW_MARGIN):end + WINDOW_MARGIN] = True
    return out


def pass_snap(video, hints):
    """Snap every tracked ball. Returns (n, 4): cx, cy, side, ambiguous; NaN where no snap."""
    px, py, seen = hints["px"], hints["py"], hints["seen"]
    snapped = np.full((len(seen), 4), np.nan, dtype=np.float32)
    cap = cv2.VideoCapture(video)
    for f in range(len(seen)):
        ok, img = cap.read()
        if not ok:
            break
        if seen[f]:
            found = snap(img, px[f], py[f])
            if found:
                snapped[f] = found
        if f % 20000 == 0:
            print("  snap pass: frame", f, flush=True)
    return snapped


def consistent(snapped):
    """True where a clean snap agrees with clean snaps on both neighbouring frames."""
    clean = ~np.isnan(snapped[:, 0]) & (snapped[:, 3] == 0)
    xy = snapped[:, :2]
    mid = (np.roll(xy, 1, 0) + np.roll(xy, -1, 0)) / 2
    ok = clean & np.roll(clean, 1) & np.roll(clean, -1)
    ok &= np.hypot(*(xy - mid).T) < CONSISTENT_PX
    ok[[0, -1]] = False
    return ok


def on_torso(snapped, feet):
    """True where the snap sits 12-60 px straight above some player's feet: a shirt number,
    shorts or a sock, not the ball. A ball at a player's feet is level with them and survives."""
    dx = np.abs(feet[:, :, 0] - snapped[:, None, 0])
    dy = feet[:, :, 1] - snapped[:, None, 1]
    return ((dx < TORSO_DX) & (dy > TORSO_DY[0]) & (dy < TORSO_DY[1])).any(1)


def tile_origin(rng, cx, cy):
    """Top-left of a 640 tile containing (cx, cy) at a random position, inside the frame."""
    def pick(c, size):
        # A ball within `margin` of the frame edge cannot have that margin inside the tile too.
        margin = 24
        lo, hi = max(0, int(c) - TILE + margin), min(size - TILE, int(c) - margin)
        return int(rng.integers(lo, max(lo, hi) + 1))
    return pick(cx, WIDTH), pick(cy, HEIGHT)


def negative_origin(rng, cx, cy):
    """Top-left of a 640 tile that does not contain (cx, cy), or None."""
    for _ in range(20):
        x0, y0 = int(rng.integers(0, WIDTH - TILE + 1)), int(rng.integers(0, HEIGHT - TILE + 1))
        if not (x0 - 20 <= cx < x0 + TILE + 20 and y0 - 20 <= cy < y0 + TILE + 20):
            return x0, y0
    return None


def pass_tiles(video, snapped, chosen, val, every_neg, seed):
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(YOLO_DIR, split, kind), exist_ok=True)
    counts = {"train": [0, 0], "val": [0, 0]}
    wanted = set(np.flatnonzero(chosen).tolist())
    last = max(wanted)
    cap = cv2.VideoCapture(video)
    for f in range(last + 1):
        ok, img = cap.read()
        if not ok or f not in wanted:
            continue
        split = "val" if val[f] else "train"
        cx, cy, side = float(snapped[f, 0]), float(snapped[f, 1]), float(snapped[f, 2])
        box = min(max(side + 4, 8), 20)
        x0, y0 = tile_origin(rng, cx, cy)
        name = "f{:06d}".format(f)
        cv2.imwrite(os.path.join(YOLO_DIR, split, "images", name + ".jpg"),
                    img[y0:y0 + TILE, x0:x0 + TILE], [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(YOLO_DIR, split, "labels", name + ".txt"), "w") as handle:
            handle.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                (cx - x0) / TILE, (cy - y0) / TILE, box / TILE, box / TILE))
        counts[split][0] += 1
        if counts[split][0] % every_neg == 0:
            origin = negative_origin(rng, cx, cy)
            if origin:
                nx, ny = origin
                cv2.imwrite(os.path.join(YOLO_DIR, split, "images", name + "_neg.jpg"),
                            img[ny:ny + TILE, nx:nx + TILE], [cv2.IMWRITE_JPEG_QUALITY, 95])
                open(os.path.join(YOLO_DIR, split, "labels", name + "_neg.txt"), "w").close()
                counts[split][1] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stride", type=int, default=3,
                        help="keep every n-th usable frame; neighbours are near-duplicates")
    parser.add_argument("--every-neg", type=int, default=3,
                        help="one ball-free tile per this many ball tiles")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resnap", action="store_true", help="redo pass 1 even if cached")
    args = parser.parse_args()

    video = require(["video"])[0]
    hints = np.load(os.path.join(OUT, "atd_ball_px.npz"))
    snap_path = os.path.join(OUT, "snapped.npz")
    if args.resnap or not os.path.isfile(snap_path):
        print("pass 1: snapping tracked balls to the image")
        np.savez_compressed(snap_path, snapped=pass_snap(video, hints))
    snapped = np.load(snap_path)["snapped"]

    n = len(snapped)
    feet = np.load(os.path.join(OUT, "players_px.npz"))["feet"]
    usable = consistent(snapped) & ~on_torso(snapped, feet) & ~excluded_frames(n)
    chosen = np.zeros(n, dtype=bool)
    chosen[np.flatnonzero(usable)[::args.stride]] = True
    val = (np.arange(n) // VAL_BLOCK) % 5 == 2
    print("tracked {}, snapped {}, consistent and outside windows {}, chosen {} ({} val)".format(
        int(hints["seen"].sum()), int((~np.isnan(snapped[:, 0])).sum()), int(usable.sum()),
        int(chosen.sum()), int((chosen & val).sum())))

    shutil.rmtree(YOLO_DIR, ignore_errors=True)
    print("pass 2: cutting tiles")
    counts = pass_tiles(video, snapped, chosen, val, args.every_neg, args.seed)
    with open(os.path.join(YOLO_DIR, "data.yaml"), "w") as handle:
        handle.write("path: {}\ntrain: train/images\nval: val/images\nnames:\n  0: ball\n".format(
            YOLO_DIR))
    print("tiles (ball, empty):", counts)


if __name__ == "__main__":
    main()
