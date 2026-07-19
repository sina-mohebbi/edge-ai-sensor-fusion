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

## Supplementary: pooling clean and noisy

I kept clean and noisy separate as you advised. To see what that costs, I also ran the
same model on all 43 recordings together, still holding out whole recordings so there is
no leakage:

| | Separate | Pooled |
|---|:---:|:---:|
| 6 apertures | 0.74 (32/43) | **0.84 (36/43)** |
| 3 levels | 0.95 (41/43) | **1.00 (43/43)** |
| Cavitation vs. no cavitation | 1.00 | 1.00 |

The gain comes mainly from the classes with few recordings: 20% and 15% improve from 3/8
to 5/8 correct, because pooling gives each of them 4 recordings instead of 2.

The two are not exactly comparable: the pooled run used 4 folds (training on about 32
recordings) while the separate runs held out one recording at a time (training on 22 and
19), so part of the gain is simply more training data.

## Supplementary: predicting the aperture as a number

Following your suggestion I also tried predicting the aperture as a continuous value
rather than choosing one of 6 classes. Being a few points off then counts as a small
error instead of a wrong class, which suits the settings that sit close together.

Run on the pooled recordings with 4 folds:

| | As 6 classes | As a number |
|---|:---:|:---:|
| 6 apertures | **0.84 (36/43)** | 0.79 (34/43) |
| Cavitation vs. no cavitation | 1.00 | 1.00 (no window wrong) |
| Average error | — | **3.8 aperture points** |

The model predicts the valve opening to within 3.8 points on average, and 33 of the 43
recordings fall within 5 points.

It fixed the 20% setting completely (4/4 against 3/4), but lost ground at nominal and 75%:
for fully open recordings it predicts around 82 to 88 instead of 100, so those snap down to
75. Predictions are pulled towards the middle of the range, which costs accuracy at the
ends. The two views are therefore complementary, and the average error is the more
informative number of the two.

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


