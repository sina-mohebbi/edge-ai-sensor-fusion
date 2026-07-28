# Cavitation detection from sound and vibration

Detecting cavitation in a centrifugal pump from its sound and vibration with a small
neural network.

A microphone and an accelerometer record the pump while an inlet valve is closed step
by step. The model reads a 1-second window of the signals and predicts the valve
aperture (`nominal, 75%, 50%, 25%, 20%, 15%`); cavitation begins around 25%.

## Main result

A **single accelerometer** turns out to be the best model here: 0.86 on the exact aperture,
above the microphone alone (0.77). Combining the two sensors does not improve on it, and no
time-frequency transform is needed. **Cavitation vs. no cavitation is detected perfectly
(100%)** on recordings the model has never seen, in both clean and noisy conditions.

The evaluation method matters more than the model: with shuffled windows every model
scores about 100%, because windows from the same recording land in both training and
test. The full write-up, with the pipeline, every model and all the results, is in
[REPORT.md](REPORT.md) (also available as `REPORT.docx`). The raw run logs are in
`results/`.

## The data

Not included here. It is about 11 GB of recordings shared separately, and it stays out of
this repository. Unzip it into `Dataset/` at the project root. Details and the label
mapping are in `Dataset/README.txt`.

## How to run

```bash
# once: turn the raw CSV files into compact arrays
python src/preprocess.py

# the honest evaluation (hold out one whole recording at a time)
python src/cross_validate.py --mode accel --condition clean
python src/cross_validate.py --mode accel --condition noisy

# pool clean + noisy together, 4 folds
python src/cross_validate.py --mode accel --condition all --folds 4

# the leaky reference (shuffle all windows, then split)
python src/shuffled_window.py --mode accel --condition clean
```

Models: `accel` (vibration only, the best here), `mic` (sound only), `earlyfusion`,
`hybrid`, `spectral`, `fusion`, `gated`. Predicting the aperture as a number:
`python src/regression.py --mode earlyfusion --condition all --folds 4`.

## The files

| File | What it does |
|------|--------------|
| `src/preprocess.py` | Reads the raw CSV files once and saves each recording compactly, plus an index with the labels. |
| `src/dataset.py` | Loads recordings and cuts them into 1-second windows of sound and vibration. |
| `src/model.py` | All the models: the single-sensor networks, the fusion variants, and the early-fusion network. |
| `src/regression.py` | Predicts the aperture as a number instead of a class. |
| `src/cross_validate.py` | The honest evaluation: hold out one recording, train on the rest, repeat for all. |
| `src/shuffled_window.py` | The leaky reference: pool all windows, shuffle, split. Kept for comparison. |
| `src/train.py` | Trains on a single split. Used early on. |
| `src/split.py` | Builds a train/val/test split and cross-validation folds by recording. |
| `results/` | Raw output of every run, and the model diagram. |

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas scipy scikit-learn matplotlib tqdm
```

Training runs on the GPU if one is available. Results are reproducible: the random seed
is fixed, so repeating a run gives the same numbers.
