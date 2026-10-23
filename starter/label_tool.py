"""Watch the tactical video and mark who has the ball.

    python starter/label_tool.py --start 11185 --end 18685 --name calib

Keys (click the video window first):
    space   play / pause
    .  ,    one frame forward / back          (while paused)
    ]  [    one second forward / back
    1       from this frame: Barcelona (H) has the ball
    2       from this frame: Athletic (A) has the ball
    3       from this frame: dead ball (D)
    u       undo the last mark
    s       slow motion on / off
    q       save and quit

Marks are saved to labels/<name>.csv after every key press, so nothing is lost if it crashes.
"""
import argparse
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import require  # noqa: E402

NAMES = {"H": "Barcelona (H)", "A": "Athletic (A)", "D": "DEAD BALL (D)"}
KEYS = {ord("1"): "H", ord("2"): "A", ord("3"): "D"}


def save(path, marks, start, end):
    with open(path, "w") as handle:
        handle.write("start,{}\nend,{}\n".format(start, end))
        for frame, label in sorted(marks):
            handle.write("{},{}\n".format(frame, label))


def load(path):
    marks = []
    if os.path.exists(path):
        for line in open(path):
            key, value = line.strip().split(",")
            if key not in ("start", "end"):
                marks.append((int(key), value))
    return marks


def current_label(marks, frame):
    label = "-"
    for f, lab in sorted(marks):
        if f <= frame:
            label = lab
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="first frame to label")
    parser.add_argument("--end", type=int, required=True, help="frame after the last one")
    parser.add_argument("--name", required=True, help="name for the labels file")
    args = parser.parse_args()

    os.makedirs(os.path.join(ROOT, "labels"), exist_ok=True)
    out_path = os.path.join(ROOT, "labels", args.name + ".csv")
    marks = load(out_path)

    video = require(["video"])[0]
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit("could not open " + video)

    frame_no = max([args.start] + [f for f, _ in marks])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, image = cap.read()
    playing, slow = False, False
    cv2.namedWindow("label", cv2.WINDOW_NORMAL)

    while True:
        if not ok:
            print("could not read frame", frame_no)
            break
        shown = cv2.resize(image, (1280, 720))
        label = current_label(marks, frame_no)
        seconds = frame_no / 25.0
        lines = [
            "frame {}   video time {}:{:05.2f}   {}".format(
                frame_no, int(seconds // 60), seconds % 60, "PLAYING" if playing else "PAUSED"),
            "now: {}".format(NAMES.get(label, "nothing marked yet")),
            "1=Barca  2=Athletic  3=Dead   space=play  ,.=frame  []=1s  u=undo  s=slow  q=quit",
        ]
        for i, text in enumerate(lines):
            cv2.putText(shown, text, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(shown, text, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("label", shown)

        key = cv2.waitKey(80 if slow else 40 if playing else 0) & 0xFF
        jump = None
        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("s"):
            slow = not slow
        elif key in KEYS:
            marks = [m for m in marks if m[0] != frame_no] + [(frame_no, KEYS[key])]
            save(out_path, marks, args.start, args.end)
            print("frame {}  {}".format(frame_no, NAMES[KEYS[key]]))
        elif key == ord("u") and marks:
            removed = sorted(marks)[-1]
            marks.remove(removed)
            save(out_path, marks, args.start, args.end)
            print("undo", removed)
        elif key == ord("."):
            jump = frame_no + 1
        elif key == ord(","):
            jump = frame_no - 1
        elif key == ord("]"):
            jump = frame_no + 25
        elif key == ord("["):
            jump = frame_no - 25

        if jump is not None:
            playing = False
            frame_no = max(0, jump)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, image = cap.read()
        elif playing:
            ok, image = cap.read()
            frame_no += 1
            if frame_no >= args.end:
                playing = False
                print("reached the end of this window")

    save(out_path, marks, args.start, args.end)
    print("saved", len(marks), "marks to", out_path)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
