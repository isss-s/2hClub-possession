# 2hClub — Technical document

**More than a Hack 2027, Phase 1: Ball Possession**
Team: 2hClub · Code: https://github.com/isss-s/2hClub-possession

---

## 1. Approach

We started from the reference baseline ("the team of the player nearest the ball has it") and
asked why it scores so low on `spellF1`, the metric that decides the ranking. `spellF1` rewards
recovering each real possession *as a whole*. The raw baseline is right on most frames but flickers
between teams, cutting every real possession into many short ones (131 predicted possessions in
the public 5 minutes, against 17 real ones).

So instead of a more complex model, we kept the nearest-player signal and added three simple,
explainable rules that give the output the *shape* of real football possession: few, long
possessions, separated by real stoppages.

1. **Gap filling.** When the ball disappears from tracking for a short time, it is usually hidden
   behind players, not out of play. Gaps of up to 2 s (50 frames) keep the team that had the ball
   before the gap. Longer gaps are treated as dead ball.
2. **Switching with patience (hysteresis).** Possession only changes team after the other team has
   been nearest to the ball for 2 s (50 frames) in a row. A passing opponent or a single touch does
   not flip possession. When a change is accepted, it is dated back to where the 2 s run began, so
   the transition is not reported late.
3. **Slow ball = dead ball.** When play is stopped (waiting for a goal kick, a free kick), the ball
   sits still or is walked to the spot. We compute the ball's speed from its tracked position frame
   to frame, take the median over a ±5 s window, and mark frames as `D` where that median is below
   3 m/s — but only for slow stretches lasting at least 12 s (300 frames). Short slow moments are
   ignored, because they are usually live play (a player stopping the ball, a slow build-up).

Why rules rather than machine learning: the only labels we have are the 5 public minutes
(17 possessions). That is far too little to train a model without overfitting, and too little even
to tune many parameters. A few rules, each motivated by how football works and each checked
against the data, are more likely to hold up on the full match.

## 2. Assets used

| asset | used? | how |
|---|---|---|
| Tactical tracking (ATD), players | yes | team of the nearest player to the ball, per frame |
| Tactical tracking (ATD), ball | yes | ball position; ball speed derived from it (the file's own ball speed field is empty) |
| Public 5-minute ground truth | yes | evaluation and analysis only, not as a training label |
| Smart Tagging | studied, not used | set-piece tags start at or near the *restart*, not at the start of the stoppage, and one of the four public dead balls has no tag at all |
| Broadcast video / tracking | no | not synchronised with the tactical timeline; left for later |
| Homography, videos | no | tracking already gives pitch coordinates |

## 3. Possession rules

- **Who has the ball:** the team of the nearest tracked player, smoothed by rules 1 and 2. No
  "contested" state: a loose ball stays with the last team until the other team has clearly
  taken over (2 s).
- **Dead ball (`D`):** (a) the ball is missing from tracking for more than 2 s, or (b) the ball has
  been slow (median under 3 m/s) for a stretch of at least 12 s.
- **Restarts:** after a dead ball, possession goes to whichever team is nearest the ball when it
  reappears / starts moving.
- **Abstention:** we never output `X`. Frames outside the match (before kick-off, half-time) are
  not graded, so their label does not matter.

## 4. Results (public 5 minutes only)

| version | spellF1 | macroF1 | possessions | mean length |
|---|---|---|---|---|
| baseline_raw | 0.038 | 0.597 | 131 | 2.0 s |
| baseline_smooth50 (official floor, 0.2433 on full match) | 0.326 | 0.621 | 26 | 10.8 s |
| baseline_smooth125 | 0.424 | 0.581 | 16 | 18.5 s |
| + gap filling, switch after 1 s | 0.318 | 0.630 | 32 | 8.3 s |
| + gap filling, switch after 2 s | 0.509 | 0.680 | 16 | 16.6 s |
| + slow-ball rule, no minimum length | 0.476 | 0.760 | 20 | 11.2 s |
| **+ slow-ball rule, minimum 12 s (submitted)** | **0.529** | **0.753** | **17** | **14.8 s** |
| ground truth | — | — | 17 | 14.2 s |

What we learned along the way:
- Changing team after 1 s still fragments possessions; 2 s matches real possession counts. 3 s
  was slightly worse.
- The slow-ball rule roughly doubled correct dead-ball frames (494 → 1,109 of 1,469 without a
  minimum length) but, applied everywhere, it broke live possessions with short false stoppages and
  lowered `spellF1`. Requiring a minimum length fixed that.
- In the public window, live stretches longer than 10 s had a median ball speed of 4.6–12 m/s; the
  two dead balls where the ball stayed visible had 2.8 and 1.7 m/s.

**Caveat.** These numbers come from 17 possessions. The scorer itself warns that differences of
about 0.04 are within noise there, and the baseline shows the trap: `smooth125` is best on the
public window but worse than `smooth50` on the full match. We therefore chose parameters by whether
the output *looks like football* (possession count and length close to reality, round values),
not only by the top score.

## 5. Limitations and what we would do next

- **Long ball-tracking gaps in live play are called dead.** In the public window this wrongly turns
  386 frames of away possession into `D`. Next: only call a long gap dead if the players also slow
  down or regroup.
- **The cautious slow-ball rule misses shorter stoppages** (quick throw-ins and free kicks under
  12 s). Next: use the Smart Tagging set-piece tags as extra evidence that a stoppage just ended.
- **The nearest player is not always in control** (aerial duels, tackles, a ball passing near a
  defender). Next: use player and ball velocity, e.g. require the ball to be close *and* moving
  with the player.
- **Restart team:** Smart Tagging names the team for most set pieces (not throw-ins); this could
  set the first possession after a stoppage directly.
- **Parameter choice** relies on a very small labelled window; a few more hand-labelled minutes,
  used only for validation, would make the choices safer.

## 6. Reproduction

Tested on macOS, Python 3.9.6. From a fresh clone of the repository, with the match data in `data/`:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-2hclub.txt
    python starter/my_second.py --min-dead 300
    python evaluation/validate_submission.py evaluation/results/my_second_w125_slow3.0_min300.csv
    cp evaluation/results/my_second_w125_slow3.0_min300.csv 2hClub.csv

We verified this from a fresh clone: the output is byte-for-byte identical to the submitted
`2hClub.csv`. Runtime is a few minutes on a laptop; no GPU and no extra data are needed.
