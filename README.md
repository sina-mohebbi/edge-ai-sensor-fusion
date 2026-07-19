# Cavitation detection from sound and vibration

Predicting the state of a centrifugal pump from its sound and vibration, using a small
neural network that reads both signals together.

A microphone and an accelerometer record the pump while an inlet valve is closed step by
step. The model takes a 1-second window of both signals and predicts the valve aperture:
`nominal, 75%, 50%, 25%, 20%, 15%`. Cavitation begins around the 25% setting.

## Main result

The model detects **cavitation vs. no cavitation perfectly (100%)** on recordings it has
never seen, in both quiet and noisy conditions. On the exact 6 apertures it reaches
0.70 (quiet) and 0.80 (noisy).

The evaluation method matters more than the model: with shuffled windows every model
scores about 100%, because windows from the same recording land in both training and
test. Full numbers are in [RESULTS.md](results/RESULTS.md), and a write-up is in
[REPORT.md](REPORT.md).

## The data

Not included here. It is about 11 GB of recordings shared separately, and it stays out of
this repository. Unzip it into `Dataset/` at the project root. Details and the label
mapping are in `Dataset/README.txt`.

## How to run

```bash
# once: turn the raw CSV files into compact arrays
python src/preprocess.py

# the honest evaluation (hold out one whole recording at a time)
python src/cross_validate.py --mode earlyfusion --condition clean
python src/cross_validate.py --mode earlyfusion --condition noisy

# the leaky reference (shuffle all windows, then split)
python src/shuffled_window.py --mode earlyfusion --condition clean
```

Available models: `earlyfusion`, `hybrid`, `spectral`, `fusion`, `gated`, `mic`, `accel`.

## The files

| File | What it does |
|------|--------------|
| `src/preprocess.py` | Reads the raw CSV files once and saves each recording compactly, plus an index with the labels. |
| `src/dataset.py` | Loads recordings and cuts them into 1-second windows of sound and vibration. |
| `src/model.py` | All the models, including the early-fusion network used for the final result. |
| `src/cross_validate.py` | The honest evaluation: hold out one recording, train on the rest, repeat for all. |
| `src/shuffled_window.py` | The leaky reference: pool all windows, shuffle, split. Kept for comparison. |
| `src/train.py` | Trains on a single split. Used early on. |
| `src/split.py` | Builds a train/val/test split and cross-validation folds by recording. |
| `results/` | Result logs and the results summary. |

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas scipy scikit-learn matplotlib tqdm
```

Training runs on the GPU if one is available. Results are reproducible: the random seed
is fixed, so repeating a run gives the same numbers.
