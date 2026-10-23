# 2hClub — Technical document

**More than a Hack 2027, Phase 1: Ball Possession**
Team: 2hClub · Code: https://github.com/isss-s/2hClub-possession

---

## 1. Approach

We started from the reference baseline ("the team of the player nearest the ball has it") and
asked why it scores so low on `spellF1`, the metric that decides the ranking. `spellF1` rewards
recovering each real possession *as a whole*. The raw baseline is right on many frames but
flickers between teams, cutting every real possession into many short ones.

Our method keeps the tracking as the only input and adds a few explainable rules that give the
output the *shape* of real football possession: few, long possessions separated by real stoppages.

1. **Control, not proximity.** A team is "on the ball" only when one of its players is within
   1.5 m of it. Possession changes team only after the other team has been on the ball for 1 s
   (25 frames) in a row; the change is dated back to where that run began. When nobody is on the
   ball (a pass in flight, a loose ball), the last team keeps possession.
   *Why:* in long possessions the defending team presses, so an opponent is often the nearest
   player without having the ball. Nearest-player labels flip back and forth; control does not.
2. **Ball out of play.** The ball seen more than 0.5 m outside the pitch lines is a dead ball.
3. **Ball missing.** Short gaps (up to 2 s) are the ball hidden behind players: possession
   continues. Gaps longer than 6 s are treated as dead ball.
4. **Slow ball = dead ball.** When the median ball speed over a ±5 s window stays under 3 m/s for
   at least 12 s, play is stopped (ball placed for a goal kick or free kick), so those frames are `D`.

Why rules rather than a trained model: labelled data is scarce, so every rule is motivated by how
football works and checked against data rather than fitted to it.

## 2. Validation: more than the public 5 minutes

The public ground truth covers 5 minutes (17 possessions), too little to choose between methods;
the scorer itself warns that differences of about 0.04 are noise there. So we **hand-labelled
additional 5-minute windows** of the tactical video, using only for validation (never as a
training label or a submission), as RULES.md allows.

- We wrote a small labelling tool (`starter/label_tool.py`): it plays the video with frame numbers
  and records a key press at every possession change or stoppage.
- **Calibration:** our labeller first labelled the public 5 minutes without looking at the
  answer. Agreement with the official labels was 94% of frames (spellF1 0.787), and the
  differences taught us the official conventions: short possessions count, and a dead ball lasts
  until the restart is actually kicked or thrown.
- We then labelled windows spread across the match (frames 35000–42500, 60000–67500,
  82000–89500, …) and evaluated every idea on all windows together.

This changed our conclusions. Our earlier best version scored 0.529 on the public window but only
about 0.2–0.25 on the new windows: the public window was unusually easy, and tuning on it alone
would have misled us.

## 3. Assets used

| asset | used? | how |
|---|---|---|
| Tactical tracking (ATD), players | yes | distance of each team's nearest player to the ball |
| Tactical tracking (ATD), ball | yes | position; speed derived from it (the file's ball speed field is empty) |
| Tactical video | validation only | hand-labelling extra windows |
| Public 5-minute ground truth | validation only | calibration of our labeller and scoring |
| Smart Tagging | studied, not used | set-piece tags mark the restart, not the stoppage; long "attacking" phases often contain unmarked stoppages, so the tags are unreliable as labels |
| Broadcast video / tracking | no | not synchronised with the tactical timeline |
| Homography | no | tracking already gives pitch coordinates |

## 4. Possession rules

- **Who has the ball:** the team with a player within 1.5 m of the ball; changes after 1 s of
  the other team's control. No "contested" state: a loose ball stays with the last team.
- **Dead ball (`D`):** ball outside the lines; ball missing for more than 6 s; or ball slow
  (median < 3 m/s) for at least 12 s.
- **Restarts:** after a dead ball, possession goes to the first team to control the ball.
- **Abstention:** we never output `X`; ungraded frames (before kick-off, half-time) do not matter.

## 5. Results

spellF1 per window (public = official ground truth; w1–w3 = our own labels):

| version | public | w1 | w2 | w3 | all windows |
|---|---|---|---|---|---|
| baseline_smooth50 (official floor) | 0.326 | 0.188 | — | — | — |
| nearest player + gap filling + 2 s switching + slow ball (`my_second`) | 0.529 | 0.249 | 0.243 | 0.207 | 0.315 |
| **control within 1.5 m + out of play + gaps + slow ball (`my_third`, submitted)** | **0.568** | **0.452** | **0.600** | **0.300** | **0.474** |

Nearby settings (on-the-ball radius 1.5–2 m, switch after 0.6–1 s) all score within about 0.03
of each other overall, so the result does not depend on one lucky setting.

## 6. Limitations and what we would do next

- **Long ball-tracking gaps in live play.** In some stretches the ball is visible only ~40% of the
  time, and gaps over 6 s are called dead even when play continues. Players' movement separates
  the two well (median team speed ~2.2 m/s in live play vs ~1.0 m/s in dead ball); using it to
  decide long gaps improved our validation score slightly and is our next step.
- **Very short possessions** (under 1 s) are deliberately ignored by the 1 s switching rule; the
  official labels count them.
- **Duels and aerial balls:** the 1.5 m radius cannot tell a tackle from control. Ball and player
  velocities could help.
- **Validation size:** a few hand-labelled windows are still a small sample.

## 7. Reproduction

Tested on macOS, Python 3.9.6. From a fresh clone of the repository, with the match data in `data/`:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-2hclub.txt
    python starter/my_third.py
    python evaluation/validate_submission.py evaluation/results/my_third.csv
    cp evaluation/results/my_third.csv 2hClub.csv

Runtime is under a minute on a laptop; no GPU and no extra data are needed.
