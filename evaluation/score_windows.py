"""Score submissions on the public window and on our hand-labelled windows, in one table.

    python evaluation/score_windows.py                      # every CSV in evaluation/results
    python evaluation/score_windows.py my_fourth my_fifth   # only these (name or path)
    python evaluation/score_windows.py --bootstrap 1000     # add a 90% interval on the all-windows score

The hand labels are marks written by starter/label_tool.py: labels/<name>.csv holds a start and
an end frame, then one "frame,label" row per change. A mark holds until the next one or the end of
the window. Frames before the first mark are left ungraded. calib.csv is our labeller's blind
attempt at the public window, so it is not used as ground truth here.

"own" scores all our windows together as one ground truth (each window its own run, so nothing
is matched across windows). "all" adds the public window. Decide on these; the public window alone is shown beside them
but is too short and too easy to choose between methods.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import possession_metrics as pm  # noqa: E402

N_FRAMES = 199319
LABELS_DIR = os.path.join(ROOT, "labels")
RESULTS_DIR = os.path.join(HERE, "results")
PUBLIC_GT = os.path.join(HERE, "gt", "BAR-ATH_possession_gt_public.csv")
NOT_GROUND_TRUTH = {"calib"}


def read_marks(path):
    """(start, end, [(frame, label), ...]) from a label_tool file."""
    start = end = None
    marks = []
    with open(path) as handle:
        for line in handle:
            key, value = line.strip().split(",")
            if key == "start":
                start = int(value)
            elif key == "end":
                end = int(value)
            else:
                marks.append((int(key), value.strip().upper()))
    return start, end, sorted(marks)


def window_gt(path):
    """Full-length gt codes for one window: X everywhere except [start, end)."""
    start, end, marks = read_marks(path)
    codes = np.full(N_FRAMES, pm.X, dtype=np.int8)
    for (frame, label), following in zip(marks, marks[1:] + [(end, None)]):
        lo, hi = max(frame, start), min(following[0], end)
        if hi > lo:
            codes[lo:hi] = pm.LABELS.index(label)
    return codes


def pool(windows):
    """One gt array holding every window in `windows`."""
    pooled = np.full(N_FRAMES, pm.X, dtype=np.int8)
    for codes in windows:
        graded = codes != pm.X
        pooled[graded] = codes[graded]
    return pooled


def load_windows():
    """{name: gt codes}: public, our windows in name order, then 'own' (ours pooled) and 'all'
    (ours plus public pooled, the "all windows" figure in 2hClub.md)."""
    windows = {"public": pm.label_codes(pm.read_label_csv(PUBLIC_GT))}
    for path in sorted(glob.glob(os.path.join(LABELS_DIR, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name not in NOT_GROUND_TRUTH:
            windows[name] = window_gt(path)
    ours = [codes for name, codes in windows.items() if name != "public"]
    windows["own"] = pool(ours)
    windows["all"] = pool(ours + [windows["public"]])
    return windows


def resolve(names):
    if not names:
        return sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv")))
    paths = []
    for name in names:
        path = name if os.path.isfile(name) else os.path.join(RESULTS_DIR, name + ".csv")
        if not os.path.isfile(path):
            sys.exit("no such submission: {}".format(name))
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="*", help="names in evaluation/results, or paths")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="block-bootstrap draws for an interval on the all-windows score")
    args = parser.parse_args()

    windows = load_windows()
    names = list(windows)
    rows = []
    for path in resolve(args.submissions):
        table = pm.read_label_csv(path)
        reason = pm.validate_submission(table, N_FRAMES)
        if reason:
            print("skipped {}: {}".format(os.path.basename(path), reason))
            continue
        pred = pm.label_codes(table)
        scores = {}
        for name, gt in windows.items():
            n = args.bootstrap if name == "all" else 0
            scores[name] = pm.evaluate(gt, pred, bootstrap_n=n)
        rows.append((os.path.splitext(os.path.basename(path))[0], scores))

    rows.sort(key=lambda row: -row[1]["all"]["spell_f1_mean"])
    width = max([len(r[0]) for r in rows] + [10])
    print("spellF1".ljust(width), *("{:>7}".format(n) for n in names), " macroF1(all)")
    for label, scores in rows:
        cells = ["{:7.3f}".format(scores[n]["spell_f1_mean"]) for n in names]
        line = "{}  {}  {:.3f}".format(label.ljust(width), " ".join(cells),
                                       scores["all"]["macro_f1"])
        if args.bootstrap:
            line += "  all 90% [{:.3f}, {:.3f}]".format(
                scores["all"]["spell_bootstrap_low"], scores["all"]["spell_bootstrap_high"])
        print(line)


if __name__ == "__main__":
    main()
