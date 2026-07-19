# regression.py
# Predict the valve aperture as a number instead of choosing one of 6 classes.
#
#   python src/regression.py --mode earlyfusion --condition all --folds 4
#
# The model outputs a single value (the aperture, where nominal counts as 100% open).
# Being off by a little is then a small error rather than a wrong class, which suits the
# settings that sit close together (20 and 15).
#
# Reported: the average error in aperture points, plus the accuracy you get by snapping
# each prediction to the nearest real aperture, so it can be compared with the classifier.

import argparse
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from dataset import CavitationWindows
from model import (FusionCNN, GatedFusionCNN, SingleModalityCNN,
                   SpectralFusionCNN, HybridFeatureCNN, EarlyFusionCNN)
from cross_validate import (accel_axes_for, forward, report_grouping, CLASS_NAMES,
                            CAV_MAP, CAV_NAMES, LEVEL_MAP, LEVEL_NAMES, PROCESSED, SEED)

# class index -> how far the valve is open, in percent
APERTURE = np.array([100.0, 75.0, 50.0, 25.0, 20.0, 15.0], dtype=np.float32)
SCALE = 100.0          # keep the target near 0..1 while training


def make_model(mode):
    if mode == "fusion":
        return FusionCNN(n_classes=1)
    if mode == "gated":
        return GatedFusionCNN(n_classes=1)
    if mode == "spectral":
        return SpectralFusionCNN(n_classes=1)
    if mode == "hybrid":
        return HybridFeatureCNN(n_classes=1)
    if mode == "earlyfusion":
        return EarlyFusionCNN(n_classes=1)
    return SingleModalityCNN(1 if mode == "accel" else 2, n_classes=1)


def train_fold(mode, train_folders, epochs, batch, device):
    train_ds = CavitationWindows(folders=train_folders, overlap=0.5,
                                 accel_axes=accel_axes_for(mode), augment=True)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)

    model = make_model(mode).to(device)
    criterion = nn.SmoothL1Loss()          # less thrown off by the odd bad window than MSE
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    targets = torch.tensor(APERTURE / SCALE, device=device)

    for _ in range(epochs):
        model.train()
        for accel, mic, y in train_dl:
            wanted = targets[y.to(device)]
            out = forward(model, mode, accel, mic, device).squeeze(1)
            loss = criterion(out, wanted)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


@torch.no_grad()
def predict(model, mode, loader, device):
    model.eval()
    trues, values = [], []
    for accel, mic, y in loader:
        out = forward(model, mode, accel, mic, device).squeeze(1)
        values.append(out.cpu().numpy() * SCALE)      # back to aperture percent
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(values)


def nearest_class(values):
    # snap a predicted aperture to the closest real setting
    return np.abs(values[:, None] - APERTURE[None, :]).argmin(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fusion", "gated", "spectral", "hybrid",
                                       "earlyfusion", "accel", "mic"],
                    default="earlyfusion")
    ap.add_argument("--condition", choices=["clean", "noisy", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--folds", type=int, default=4,
                    help="0 = leave one recording out; N > 0 = N folds over recordings")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    index = pd.read_csv(PROCESSED / "index.csv")
    if args.condition != "all":
        index = index[index["condition"] == args.condition]
    recs = index.reset_index(drop=True)
    n = len(recs)
    axes = accel_axes_for(args.mode)

    if args.folds > 0:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
        fold_list = [(list(recs.loc[tr, "folder"]), list(recs.loc[te, "folder"]))
                     for tr, te in skf.split(recs, recs["label_idx"])]
        how = f"{args.folds} folds over recordings"
    else:
        fold_list = [([recs.loc[j, "folder"] for j in range(n) if j != i],
                      [recs.loc[i, "folder"]]) for i in range(n)]
        how = "leave one recording out"

    print(f"device: {device} | mode: {args.mode} | condition: {args.condition} "
          f"| predicting aperture as a number | {how} "
          f"({n} recordings, {len(fold_list)} trainings)\n")

    errors, all_true, all_pred = [], [], []
    rec_true, rec_pred = [], []
    done = 0
    for train_folders, test_folders in fold_list:
        model = train_fold(args.mode, train_folders, args.epochs, args.batch, device)

        for test_folder in test_folders:
            test_ds = CavitationWindows(folders=[test_folder], overlap=0.0, accel_axes=axes)
            test_dl = DataLoader(test_ds, batch_size=args.batch, num_workers=0)
            labels, values = predict(model, args.mode, test_dl, device)

            true_label = int(labels[0])
            true_open = APERTURE[true_label]
            guessed_open = float(values.mean())          # one number per recording
            snapped = int(np.abs(guessed_open - APERTURE).argmin())

            errors.append(abs(guessed_open - true_open))
            all_true.append(labels)
            all_pred.append(nearest_class(values))
            rec_true.append(true_label)
            rec_pred.append(snapped)

            done += 1
            mark = "ok  " if snapped == true_label else "MISS"
            print(f"[{done:2d}/{n}] {test_folder}  true={true_open:5.1f}% "
                  f"predicted={guessed_open:6.1f}%  -> {CLASS_NAMES[snapped]:>7} {mark}")
            del test_ds, test_dl

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    errors = np.array(errors)
    yt, pt = np.concatenate(all_true), np.concatenate(all_pred)
    rec_true, rec_pred = np.array(rec_true), np.array(rec_pred)

    print(f"\n===== {args.mode} / {args.condition}: aperture as a number, {how} =====")
    print(f"recordings                    : {n}")
    print(f"average error                 : {errors.mean():.1f} aperture points "
          f"(+/- {errors.std():.1f})")
    print(f"largest error                 : {errors.max():.1f} points")
    print(f"within 5 points of the truth  : {(errors <= 5).sum()}/{n}")
    print(f"accuracy after snapping       : {(rec_true == rec_pred).sum()}/{n} "
          f"= {(rec_true == rec_pred).mean():.3f}")

    print("\nrecording confusion after snapping (rows = true, cols = predicted):")
    print("order:", CLASS_NAMES)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(rec_true, rec_pred, labels=range(6)))

    report_grouping("cavitation vs not", CAV_MAP, CAV_NAMES, yt, pt, rec_true, rec_pred)
    report_grouping("severity level", LEVEL_MAP, LEVEL_NAMES, yt, pt, rec_true, rec_pred)


if __name__ == "__main__":
    main()
