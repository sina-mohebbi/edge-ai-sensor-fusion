# Cavitation detection from sound and vibration — progress report

## Summary

I built a model that predicts the pump's valve aperture from the microphone and
accelerometer signals, and evaluated it in two different ways. The main finding is that
the evaluation method changes the result far more than the model does: when 1-second
windows are shuffled and split, every model reaches close to 100%; when whole recordings
are held out for testing, the honest accuracy is 0.70–0.80 across the 6 apertures, while
cavitation vs. no cavitation is still detected perfectly.

## Setup

- Clean and noisy signals kept as two separate datasets, as you advised.
- Target: the 6 valve apertures (nominal, 75%, 50%, 25%, 20%, 15%).
- Signals cut into 1-second windows: microphone (2 channels, 48 kHz) and accelerometer
  (3 axes, 4 kHz).
- 23 clean recordings, 20 noisy. One recording (`20260401_133648`) was excluded because
  its accelerometer file is empty.

## Evaluation

Two protocols were compared:

1. **Shuffled windows (70/15/15).** All windows are pooled and split at random. Windows
   from the same recording end up in both training and test, so the scores are optimistic.
2. **New recordings (leave one recording out).** One whole recording is held out, the
   model trains on all the others, and this repeats until every recording has been the
   test recording once. Training and test never share a recording.

Since each recording is a single operating point, the second protocol measures whether
the model generalises to a recording it has never seen.

## Model

An early-fusion network. The microphone and accelerometer signals are converted to
spectrograms, resized to a common grid, stacked together as channels of a single image,
and processed by one small CNN (about 115k parameters). The two signals are therefore
combined at the input rather than in separate branches.

Training: AdamW, learning rate 1e-3, weight decay 1e-4, cosine schedule, batch size 32,
20 epochs, class weighting, fixed random seed so results are reproducible.

## Results

**Shuffled windows (clean):**

| Model | Accuracy |
|-------|:---:|
| Reference fusion | 0.997 |
| Sound only | 0.999 |
| Hybrid | 1.000 |
| **Early fusion** | **1.000** |

**New recordings (early fusion):**

| Condition | 6 apertures | 3 levels | Cavitation vs. no cavitation |
|-----------|:---:|:---:|:---:|
| Clean | 0.70 (16/23) | 0.96 (22/23) | **1.00 (23/23)** |
| Noisy | 0.80 (16/20) | 0.95 (19/20) | **1.00 (20/20)** |

*3 levels = none (nominal, 75%, 50%) / onset (25%) / developing (20%, 15%).*

## Error analysis

Every misclassification is between neighbouring, physically similar settings:

- **nominal vs. 75%** — both without cavitation, differing only in flow rate.
- **20% vs. 15%** — both in the developing region.

No recording with cavitation was ever classified as no cavitation, in either condition.

## Limitations

- 20% and 15% have only 2 recordings each per condition, so holding one out leaves a
  single training example. Their individual accuracy is therefore unreliable, although
  they are reliable when reported together as one level.
- nominal and 75% look almost identical in the signals, which is consistent with
  cavitation not being present at those settings.
- Augmentation, larger spectrograms, and different training settings were all tested and
  none improved the 6-aperture result, which suggests the limit is the amount of data
  rather than the model.

## Possible next steps

- More recordings of 20% and 15% would be the most useful addition.
- Predicting severity as a continuous value instead of 6 separate classes.
- Two-stage classification (first the level, then the aperture within it) and
  self-supervised pretraining.

## Question

Would you prefer the main reported result to use the new-recording protocol, or the
shuffled-window one so it is directly comparable with the existing results?
