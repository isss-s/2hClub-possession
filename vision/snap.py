"""Find the real ball near where the tracking says it is.

The projected tracking ball is a search hint, not a label: it misses the ball by 6-20 px on the
ground and by 40-60 px when the ball is in the air (the tracking reports the ground point under
it), and the ball itself is only ~6 px across. snap() looks inside a window around the hint for a
small, bright, low-saturation, compact blob and returns the most ball-like one.

Classical on purpose: it only has to be right on the easy frames, because those become the
detector's first training set. The detector then finds the hard ones.
"""
import cv2
import numpy as np

SEARCH_RADIUS = 70          # px around the hint; covers lofted balls
TOPHAT_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
MIN_CONTRAST = 35           # brightness above the local background (top-hat), 0-255
MIN_AREA, MAX_AREA = 5, 110
MAX_SIDE = 14
MAX_SATURATION = 90
MIN_GRASS = 0.6             # share of the ring around the blob that must be grass
RING = (7, 13)              # inner and outer radius of that ring, px


def grass_around(hsv, centre):
    """Share of pixels in a ring around `centre` that are grass green."""
    h, w = hsv.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(xx - centre[0], yy - centre[1])
    ring = (dist >= RING[0]) & (dist <= RING[1])
    if ring.sum() < 50:
        return 0.0                                     # too close to the crop edge to tell
    hue, sat = hsv[..., 0][ring], hsv[..., 1][ring]
    return float(((hue >= 30) & (hue <= 90) & (sat >= 35)).mean())


def candidates(img, x, y, radius=SEARCH_RADIUS):
    """Ball-like blobs within `radius` of (x, y): list of (cx, cy, contrast, area, side)."""
    h, w = img.shape[:2]
    x0, y0 = int(max(0, x - radius)), int(max(0, y - radius))
    x1, y1 = int(min(w, x + radius)), int(min(h, y + radius))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return []
    crop = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2]
    tophat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, TOPHAT_KERNEL)
    mask = (tophat > MIN_CONTRAST).astype(np.uint8)
    n, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
    found = []
    for i in range(1, n):
        bw, bh, area = stats[i, 2], stats[i, 3], stats[i, 4]
        if not (MIN_AREA <= area <= MAX_AREA) or max(bw, bh) > MAX_SIDE:
            continue
        if max(bw, bh) > 2.2 * min(bw, bh) or area < 0.35 * bw * bh:
            continue                                   # line fragments, not round
        blob = labels == i
        if hsv[..., 1][blob].mean() > MAX_SATURATION:
            continue                                   # coloured kit, not a white ball
        if grass_around(hsv, centres[i]) < MIN_GRASS:
            continue                                   # socks and shirt numbers sit on a player
        found.append((centres[i, 0] + x0, centres[i, 1] + y0, float(tophat[blob].max()),
                      int(area), int(max(bw, bh))))
    return found


def snap(img, x, y, radius=SEARCH_RADIUS):
    """(cx, cy, side, ambiguous) of the most ball-like blob near the hint, or None.

    ambiguous is True when a second blob is nearly as good, which the dataset builder treats as
    "not sure" and skips.
    """
    found = candidates(img, x, y, radius)
    if not found:
        return None
    scored = sorted(((c[2] - 0.4 * np.hypot(c[0] - x, c[1] - y), c) for c in found),
                    reverse=True)
    best = scored[0][1]
    ambiguous = len(scored) > 1 and scored[1][0] > scored[0][0] - 15
    return best[0], best[1], best[4], ambiguous
