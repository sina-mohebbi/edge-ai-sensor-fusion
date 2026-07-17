# Results

Every model I trained this session, in order, with the scores I got.

Two things to know first:

- The data has two kinds of recordings: **quiet** (only our pump running) and **noisy**
  (a second pump running nearby). Later on I keep these two apart, as the professor asked.
- How you measure matters a lot. I used a few different ways to score the models, so scores
  from different ways are **not** directly comparable. Higher isn't always better — sometimes
  it just means the test was easier (or leaked).

## The four ways I measured

1. **One split** — split the recordings once into train and test. Quick, but a bit lucky.
2. **Four rounds** — every recording gets tested once, averaged over four rounds. Steadier.
3. **New recording test** — test on a recording the model never saw, and repeat for every
   recording. The most honest way. I run it for quiet and noisy separately.
4. **Shuffled pieces test** — cut every recording into 1 second pieces, shuffle them all
   together, then split. It cheats a little (pieces of the same recording land in both train
   and test), so scores look great but aren't real.

---

## Phase 1 — first models, one split (quiet + noisy mixed)

The fusion model reached about **0.90** while tuning, then dropped to about **0.66** on the real
test — the first hint that the easy scores were misleading.

| Model | Accuracy | Macro F1 |
|-------|:---:|:---:|
| Sound only | 0.812 | 0.759 |
| Fusion (sound + vibration) | 0.703 | 0.626 |
| Gated fusion | 0.695 | 0.621 |
| Vibration only | 0.626 | 0.519 |

Surprise: **sound alone beat the fusion.** The weak vibration branch was dragging fusion down.

## Phase 2 — four rounds (quiet + noisy mixed)

| Model | Accuracy | Macro F1 |
|-------|:---:|:---:|
| Fusion | 0.578 (± 0.098) | 0.530 (± 0.084) |
| Spectral (frequency features) | 0.608 (± 0.081) | 0.550 (± 0.138) |

Adding frequency features helped the real cavitation classes.

## Phase 3 — new recording test, quiet and noisy apart (the honest one)

| Setting | Model | Accuracy | Macro F1 | Recordings right |
|---------|-------|:---:|:---:|:---:|
| Quiet | Sound only | 0.593 | 0.525 | 15 of 23 |
| Quiet | Hybrid (all features) | **0.705** | **0.627** | **17 of 23** |
| Noisy | Hybrid | *to be re-run* | | |
| Noisy | Sound only | *to be re-run* | | |

The **hybrid** model, using all the extra features, beats sound alone by about **10 points**.
This is the honest result to report.

## Phase 4 — shuffled pieces test (the leaky way), quiet

| Model | Accuracy | Macro F1 |
|-------|:---:|:---:|
| Fusion | 0.997 | 0.997 |
| Sound only | 0.999 | 0.999 |
| Hybrid | 1.000 | 1.000 |

Every model scores nearly **100%** here. This shows the high scores come from the leaky way of
measuring, not from a better model.

---

## What it all means

- Measured honestly (new recordings), the model reaches about **0.70** on quiet data. Real and
  trustworthy, even if it looks lower.
- Measured the leaky way (shuffled pieces), every model hits about **100%**, which shows those
  high numbers are inflated by leakage.
- The extra features (sound + vibration together, all three vibration directions, frequency
  content, and a few simple numbers like loudness and spikiness) genuinely help on the honest test.
- The model is good at spotting **cavitation** (25% valve and below). It struggles with **nominal
  vs 75%** (both have no cavitation, so they look alike) and with **20%** (only 2 recordings, too
  few to learn from).
