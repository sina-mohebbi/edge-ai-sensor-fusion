# cross_validate.py
# Leave-one-recording-out cross-validation, run separately for the clean and the
# noisy dataset (per the professor: clean and noisy are two distinct datasets).
#
#   python src/cross_validate.py --mode spectral --condition clean
#   python src/cross_validate.py --mode spectral --condition noisy
#
# Why leave-one-recording-out: once we split by condition, the severe classes have
# only 2 recordings each — too few for k-fold. So we train on every recording but
# one and test on the held-out recording, repeating for all of them. Each recording
# is tested exactly once, on a model that never saw it, using almost all the data
# for training. We report per-recording accuracy (mean +/- std), pooled window
# metrics, and a recording-level majority-vote accuracy + confusion matrix.
#
# No separate validation set: each fold trains for a fixed number of epochs,
# applied identically to every model so comparisons stay fair.

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dataset import CavitationWindows
from model import (FusionCNN, GatedFusionCNN, SingleModalityCNN,
                   SpectralFusionCNN, HybridFeatureCNN)

HERE = Path(__file__).resolve().parent.parent
PROCESSED = HERE / "data" / "processed"
CLASS_NAMES = ["nominal", "75", "50", "25", "20", "15"]
SEED = 42

# modes that take both modalities and use all 3 accel axes
BOTH_MODALITY = ("fusion", "gated", "spectral", "hybrid")
ALL_AXES = ("spectral", "hybrid")


def accel_axes_for(mode):
    return (0, 1, 2) if mode in ALL_AXES else (2,)


def make_model(mode):
    if mode == "fusion":
        return FusionCNN()
    if mode == "gated":
        return GatedFusionCNN()
    if mode == "spectral":
        return SpectralFusionCNN()
    if mode == "hybrid":
        return HybridFeatureCNN()
    return SingleModalityCNN(1 if mode == "accel" else 2)


def forward(model, mode, accel, mic, device):
    accel, mic = accel.to(device), mic.to(device)
    if mode in BOTH_MODALITY:
        return model(accel, mic)
    return model(accel if mode == "accel" else mic)


def class_weights(ds, device):
    labels = np.array([w[1] for w in ds.windows])
    counts = np.bincount(labels, minlength=6)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model, mode, loader, device):
    model.eval()
    trues, preds = [], []
    for accel, mic, y in loader:
        out = forward(model, mode, accel, mic, device)
        preds.append(out.argmax(1).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def train_fold(mode, train_folders, epochs, batch, device):
    train_ds = CavitationWindows(folders=train_folders, overlap=0.5,
                                 accel_axes=accel_axes_for(mode))
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)

    model = make_model(mode).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        model.train()
        for accel, mic, y in train_dl:
            y = y.to(device)
            loss = criterion(forward(model, mode, accel, mic, device), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["fusion", "gated", "spectral", "hybrid", "accel", "mic"],
                    default="hybrid")
    ap.add_argument("--condition", choices=["clean", "noisy", "all"], default="clean")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    index = pd.read_csv(PROCESSED / "index.csv")
    if args.condition != "all":
        index = index[index["condition"] == args.condition]
    recs = index.reset_index(drop=True)
    n = len(recs)
    axes = accel_axes_for(args.mode)
    print(f"device: {device} | mode: {args.mode} | condition: {args.condition} "
          f"| leave-one-recording-out ({n} recordings)\n")

    per_rec_acc, all_true, all_pred, rec_true, rec_pred = [], [], [], [], []
    for i in range(n):
        test_folder = recs.loc[i, "folder"]
        train_folders = [recs.loc[j, "folder"] for j in range(n) if j != i]

        model = train_fold(args.mode, train_folders, args.epochs, args.batch, device)

        test_ds = CavitationWindows(folders=[test_folder], overlap=0.0, accel_axes=axes)
        test_dl = DataLoader(test_ds, batch_size=args.batch, num_workers=0)
        yt, pt = predict(model, args.mode, test_dl, device)

        acc = accuracy_score(yt, pt)
        true_label = int(yt[0])
        pred_label = int(np.bincount(pt, minlength=6).argmax())     # majority vote
        per_rec_acc.append(acc)
        all_true.append(yt)
        all_pred.append(pt)
        rec_true.append(true_label)
        rec_pred.append(pred_label)
        mark = "ok  " if pred_label == true_label else "MISS"
        print(f"[{i+1:2d}/{n}] {test_folder}  true={CLASS_NAMES[true_label]:>7} "
              f"pred={CLASS_NAMES[pred_label]:>7}  {mark} | window acc {acc:.3f}")

        del model, test_ds, test_dl
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    per_rec_acc = np.array(per_rec_acc)
    yt, pt = np.concatenate(all_true), np.concatenate(all_pred)
    rec_true, rec_pred = np.array(rec_true), np.array(rec_pred)

    print(f"\n===== {args.mode} / {args.condition}: leave-one-recording-out =====")
    print(f"recordings                    : {n}")
    print(f"per-recording window accuracy : {per_rec_acc.mean():.3f} +/- {per_rec_acc.std():.3f}")
    print(f"pooled window accuracy        : {accuracy_score(yt, pt):.3f}")
    print(f"pooled window macro-F1        : {f1_score(yt, pt, average='macro'):.3f}")
    print(f"recording-level accuracy      : {(rec_true == rec_pred).sum()}/{n} "
          f"= {(rec_true == rec_pred).mean():.3f}  (majority vote per recording)")

    print("\npooled window confusion matrix (rows = true, cols = pred):")
    print("order:", CLASS_NAMES)
    print(confusion_matrix(yt, pt, labels=range(6)))
    print("\nrecording-level confusion matrix (majority vote):")
    print(confusion_matrix(rec_true, rec_pred, labels=range(6)))


if __name__ == "__main__":
    main()
