# Results

All numbers below use the same task: predict the valve aperture from a 1-second window of
sound and vibration. Clean and noisy are kept as two separate datasets.

## The two ways of measuring

**Shuffled windows.** All windows are pooled, shuffled, and split 70/15/15. Windows from
the same recording end up in both training and test, so the scores come out near perfect.
Kept only as a reference point.

**New recordings.** One whole recording is held out, the model trains on all the others,
and this repeats until every recording has been the test recording once. Training and test
never share a recording, so this measures real generalisation.

## Final model: early fusion

Sound and vibration are turned into spectrograms, resized to a common grid, stacked as
channels of one image, and read by a single small CNN (~115k parameters). Both signals are
combined at the input rather than in separate branches.

Training: AdamW, lr 1e-3, weight decay 1e-4, cosine schedule, batch 32, 20 epochs, class
weighting, fixed seed (results are reproducible).

### New recordings (the honest result)

| Condition | 6 apertures | 3 levels | Cavitation vs. none |
|-----------|:---:|:---:|:---:|
| Clean | 0.70 (16/23) | 0.96 (22/23) | **1.00 (23/23)** |
| Noisy | 0.80 (16/20) | 0.95 (19/20) | **1.00 (20/20)** |

*3 levels = none (nominal, 75%, 50%) / onset (25%) / developing (20%, 15%).*

### Shuffled windows (the leaky reference, clean)

| Model | Accuracy |
|-------|:---:|
| Reference fusion | 0.997 |
| Sound only | 0.999 |
| Hybrid | 1.000 |
| Early fusion | 1.000 |

Every model reaches about 100% here. The same early-fusion model scores 1.00 with shuffled
windows and 0.70 with held-out recordings, which shows the split matters more than the
model.

## Other models tried

Explored with the same held-out-recording method, before the seed was fixed, so these are
indicative rather than exactly reproducible:

| Model | Clean, 6 apertures |
|-------|:---:|
| Sound only | 0.65 |
| Hybrid (raw + spectrogram + simple descriptors) | 0.74 |
| Early fusion | 0.70 (seeded, final) |

Sound alone was consistently stronger than vibration alone. Simple fusion of the two was
often *worse* than sound alone, because the weaker vibration branch pulled the result down.
Early fusion, which mixes the two at the input, avoided that and is the smallest model of
the group.

## Where the errors are

Every mistake is between neighbouring, physically similar settings:

- **nominal vs. 75%** — neither has cavitation, they differ only in flow rate.
- **20% vs. 15%** — both in the developing region.

No recording with cavitation was ever labelled as no cavitation, in either condition.

## Limits

- 20% and 15% have only **2 recordings each** per condition, so holding one out leaves a
  single training example. Their individual scores are unreliable, though they are reliable
  when reported together as one level.
- nominal and 75% look almost identical in the signals, consistent with cavitation not
  being present at those settings.
- Augmentation, larger spectrograms (128 instead of 64), soft voting, and different
  training settings were all tested. None changed the 6-aperture result, which points to
  the amount of data being the limit rather than the model.
