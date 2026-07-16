# split.py
# Builds two things, both at the RECORDING level and both stratified by class:
#   1) split.csv  — a 70/15/15 train/val/test split, for developing the pipeline
#   2) folds.csv  — 4 cross-validation folds, for the final, reliable numbers
#
# Recording-level means a whole recording goes into exactly one set. Windows cut
# from the same 180 s recording are nearly identical, so letting any leak from
# train into test would fake the score (that's the flaw in the original paper).
# We split recordings here; the windows get made later, inside whichever set the
# recording ended up in.
#
# k = 4 folds is the largest that still puts one recording of EVERY class in every
# fold, because the rarest classes (20% and 15%) have only 4 recordings each. It's
# the natural number for this dataset.

from pathlib import Path
import random

import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent.parent
PROCESSED = HERE / "data" / "processed"
INDEX = PROCESSED / "index.csv"

SEED = 42
K_FOLDS = 4
VAL_FRAC = 0.15
TEST_FRAC = 0.15
CLASS_ORDER = ["nominal", "75", "50", "25", "20", "15"]


def single_split(index, seed):
    # Go class by class so every class shows up in all three sets — even the
    # 4-recording ones, which end up as 2 train / 1 val / 1 test.
    rng = random.Random(seed)
    split_of = {}
    for _, rows in index.groupby("label_idx"):
        folders = list(rows["folder"])
        rng.shuffle(folders)
        n = len(folders)
        n_test = max(1, round(TEST_FRAC * n))
        n_val = max(1, round(VAL_FRAC * n))
        for f in folders[:n_test]:
            split_of[f] = "test"
        for f in folders[n_test:n_test + n_val]:
            split_of[f] = "val"
        for f in folders[n_test + n_val:]:
            split_of[f] = "train"
    return split_of


def cv_folds(index, k, seed):
    # Stratified k-fold over the recordings; each recording is the test set once.
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_of = {}
    for fold, (_, test_idx) in enumerate(skf.split(index, index["label_idx"])):
        for i in test_idx:
            fold_of[index.iloc[i]["folder"]] = fold
    return fold_of


def main():
    index = pd.read_csv(INDEX).sort_values("folder").reset_index(drop=True)

    index["split"] = index["folder"].map(single_split(index, SEED))
    index["fold"] = index["folder"].map(cv_folds(index, K_FOLDS, SEED))

    index[["folder", "split"]].to_csv(PROCESSED / "split.csv", index=False)
    index[["folder", "fold"]].to_csv(PROCESSED / "folds.csv", index=False)

    # ---- print everything so we can eyeball that nothing's empty or lopsided ----
    label = pd.Categorical(index["label"].astype(str), categories=CLASS_ORDER, ordered=True)

    print("Single split (phase 1) — recordings per class:")
    print(pd.crosstab(label, index["split"])[["train", "val", "test"]].to_string())

    print("\nCross-validation (phase 2) — recordings per class in each fold:")
    print(pd.crosstab(label, index["fold"]).to_string())

    print("\nSanity check — clean/noisy and distance spread across the split:")
    print(pd.crosstab(index["condition"], index["split"]).to_string())
    print(pd.crosstab(index["distance_cm"], index["split"]).to_string())


if __name__ == "__main__":
    main()