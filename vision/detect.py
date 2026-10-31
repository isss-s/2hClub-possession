"""Run the ball detector over a span of the tactical video.

    python vision/detect.py --start 82000 --end 83000          # writes balldet/det_82000_83000.npz
    python vision/detect.py --weights balldet/best.pt --device cpu

The detector was trained on 640x640 full-resolution tiles (the ball is ~6 px wide), so a frame is
cut into 8 overlapping 640 tiles -- 4 across, 2 down -- rather than shrunk. Overlap is ~210 px
across and 200 px down, so a ball on a tile edge is whole in the neighbouring tile. Hits from
different tiles within MERGE_PX of each other are the same ball and keep the best score.

Output, per frame, the top MAX_PER_FRAME candidates (x, y, conf), NaN-padded. Keeping several
low-confidence candidates instead of one answer is deliberate: the temporal tracker that comes
next decides which one is the ball. Like everything in balldet/, it is derived data: never commit.
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import require  # noqa: E402

TILE = 640
TILE_X = (0, 427, 853, 1280)
TILE_Y = (0, 440)
MIN_CONF = 0.05
MERGE_PX = 8
MAX_PER_FRAME = 5


def merge(points):
    """points: list of (x, y, conf). Greedy merge of hits closer than MERGE_PX, best first."""
    kept = []
    for x, y, conf in sorted(points, key=lambda p: -p[2]):
        if all(np.hypot(x - kx, y - ky) >= MERGE_PX for kx, ky, _ in kept):
            kept.append((x, y, conf))
    return kept[:MAX_PER_FRAME]


def detect_frame(model, img, imgsz, device):
    tiles = [img[y0:y0 + TILE, x0:x0 + TILE] for y0 in TILE_Y for x0 in TILE_X]
    origins = [(x0, y0) for y0 in TILE_Y for x0 in TILE_X]
    results = model.predict(tiles, imgsz=imgsz, conf=MIN_CONF, device=device, verbose=False)
    points = []
    for (x0, y0), result in zip(origins, results):
        boxes = result.boxes
        for (cx, cy, _, _), conf in zip(boxes.xywh.cpu().numpy(), boxes.conf.cpu().numpy()):
            points.append((float(cx + x0), float(cy + y0), float(conf)))
    return merge(points)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True, help="exclusive")
    parser.add_argument("--weights", default=os.path.join(ROOT, "balldet", "best.pt"))
    parser.add_argument("--imgsz", type=int, default=1280, help="must match training")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    out = args.out or os.path.join(ROOT, "balldet", "det_{}_{}.npz".format(args.start, args.end))
    dets = np.full((args.end - args.start, MAX_PER_FRAME, 3), np.nan, dtype=np.float32)
    cap = cv2.VideoCapture(require(["video"])[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    for i in range(args.end - args.start):
        ok, img = cap.read()
        if not ok:
            break
        for j, point in enumerate(detect_frame(model, img, args.imgsz, args.device)):
            dets[i, j] = point
        if i % 250 == 0:
            print("  frame", args.start + i, flush=True)
    np.savez_compressed(out, start=args.start, dets=dets)
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
