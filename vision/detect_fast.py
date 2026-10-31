"""Ball detector over the video, fast: the version the full-match run uses.

    python vision/detect_fast.py --start 82000 --end 82200        # local check
    vision/detect_fast_colab.ipynb runs this same file on a GPU, whole match

Same detector and the same 8 overlapping 640 tiles per frame as vision/detect.py, made fast by
doing the work on the GPU and in bulk:
  * several frames are decoded ahead on a thread while the GPU works;
  * tiles are cut, converted and upscaled 2x on the GPU (the 640 tile -> 1280 input that the model
    was trained on) instead of one by one on the CPU;
  * all tiles of BATCH_FRAMES frames go through the model in one call;
  * optionally a TensorRT engine instead of the .pt weights (same model, faster on NVIDIA).

It saves more than vision/detect.py did -- the top MAX_PER_FRAME candidates down to MIN_CONF --
so that everything downstream can change without ever running the detector again.
Output: balldet/fast/det_<start>_<end>.npz with start and dets (n, MAX_PER_FRAME, 3: x, y, conf),
NaN-padded. Derived data: never commit.
"""
import argparse
import os
import queue
import sys
import threading

import cv2
import numpy as np

TILE = 640
TILE_X = (0, 427, 853, 1280)
TILE_Y = (0, 440)
ORIGINS = [(x0, y0) for y0 in TILE_Y for x0 in TILE_X]
MIN_CONF = 0.01
MERGE_PX = 8
MAX_PER_FRAME = 10


def merge(points):
    """points: list of (x, y, conf). Greedy merge of hits closer than MERGE_PX, best first."""
    kept = []
    for x, y, conf in sorted(points, key=lambda p: -p[2]):
        if all(np.hypot(x - kx, y - ky) >= MERGE_PX for kx, ky, _ in kept):
            kept.append((x, y, conf))
    return kept[:MAX_PER_FRAME]


def read_frames(video, start, end, out_queue):
    """Decode frames start..end-1 onto `out_queue` as (index, image); None at the end."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(end - start):
        ok, img = cap.read()
        if not ok:
            break
        out_queue.put((i, img))
    out_queue.put(None)


def tiles_on_device(frames, device, imgsz):
    """BGR uint8 frames -> (len(frames) * 8, 3, imgsz, imgsz) RGB float in 0-1, frame-major."""
    import torch
    import torch.nn.functional as F
    batch = torch.from_numpy(np.stack(frames)).to(device)
    batch = batch[..., [2, 1, 0]].permute(0, 3, 1, 2).float().div_(255)
    tiles = torch.stack([batch[:, :, y0:y0 + TILE, x0:x0 + TILE] for x0, y0 in ORIGINS], 1)
    tiles = tiles.reshape(-1, 3, TILE, TILE)
    if imgsz != TILE:
        # Bilinear with half-pixel centres: what cv2.INTER_LINEAR in the training letterbox does.
        tiles = F.interpolate(tiles, size=(imgsz, imgsz), mode="bilinear", align_corners=False)
    return tiles.contiguous()


def detect_batch(model, frames, device, imgsz, half):
    tiles = tiles_on_device(frames, device, imgsz)
    results = model.predict(tiles, imgsz=imgsz, conf=MIN_CONF, half=half, device=device,
                            max_det=MAX_PER_FRAME, verbose=False)
    scale = TILE / imgsz
    out = []
    for f in range(len(frames)):
        points = []
        for k, (x0, y0) in enumerate(ORIGINS):
            boxes = results[f * len(ORIGINS) + k].boxes
            for (cx, cy, _, _), conf in zip(boxes.xywh.cpu().numpy(), boxes.conf.cpu().numpy()):
                points.append((float(cx * scale + x0), float(cy * scale + y0), float(conf)))
        out.append(merge(points))
    return out


def run_span(model, video, start, end, device, imgsz, half, batch_frames):
    """Detections for frames start..end-1: (end - start, MAX_PER_FRAME, 3), NaN-padded."""
    dets = np.full((end - start, MAX_PER_FRAME, 3), np.nan, dtype=np.float32)
    frames_q = queue.Queue(maxsize=batch_frames * 4)
    threading.Thread(target=read_frames, args=(video, start, end, frames_q), daemon=True).start()
    pending, done = [], False
    while not done:
        item = frames_q.get()
        if item is None:
            done = True
        else:
            pending.append(item)
        if pending and (done or len(pending) == batch_frames):
            found = detect_batch(model, [img for _, img in pending], device, imgsz, half)
            for (i, _), points in zip(pending, found):
                for j, point in enumerate(points):
                    dets[i, j] = point
            pending = []
    return dets


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from games import require
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True, help="exclusive")
    parser.add_argument("--weights", default=os.path.join(root, "balldet", "best.pt"))
    parser.add_argument("--imgsz", type=int, default=1280, help="must match training")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--batch-frames", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model = YOLO(args.weights)
    dets = run_span(model, require(["video"])[0], args.start, args.end, args.device, args.imgsz,
                    args.half, args.batch_frames)
    out = args.out or os.path.join(root, "balldet", "fast",
                                   "det_{}_{}.npz".format(args.start, args.end))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, start=args.start, dets=dets)
    print("wrote", os.path.relpath(out, root))


if __name__ == "__main__":
    main()
