# paper_protocol.py
# Reproduce the paper's evaluation so we can compare on its own terms.
#
# It pools all windows of one condition (clean or noisy) and splits them 70/15/15
# at the WINDOW level (shuffled frames). This has leakage — windows from the same
# recording land in both train and test — so the numbers are optimistic. We report
# it ONLY as a like-for-like comparison with the paper (and to beat it there); the
# honest number is the leave-one-recording-out result in cross_validate.py.
#
#   python src/paper_protocol.py --mode fusion --condition clean   # reproduce ~paper
#   python src/paper_protocol.py --mode hybrid --condition clean   # try to beat it

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dataset import CavitationWindows
from cross_validate import make_model, forward, accel_axes_for, CLASS_NAMES

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
SEED = 42


def predict(model, mode, loader, device):
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for accel, mic, y in loader:
            out = forward(model, mode, accel, mic, device)
            preds.append(out.argmax(1).cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["fusion", "gated", "spectral", "hybrid", "accel", "mic"],
                    default="hybrid")
    ap.add_argument("--condition", choices=["clean", "noisy", "all"], default="clean")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    index = pd.read_csv(PROCESSED / "index.csv")
    if args.condition != "all":
        index = index[index["condition"] == args.condition]
    folders = list(index["folder"])

    full = CavitationWindows(folders=folders, overlap=0.5,
                             accel_axes=accel_axes_for(args.mode))
    n = len(full)

    # shuffle all windows together and split 70/15/15  <-- this is the leaky part
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_test, n_val = int(0.15 * n), int(0.15 * n)
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    train_dl = DataLoader(Subset(full, train_idx), batch_size=args.batch,
                          shuffle=True, num_workers=0)
    val_dl = DataLoader(Subset(full, val_idx), batch_size=args.batch, num_workers=0)
    test_dl = DataLoader(Subset(full, test_idx), batch_size=args.batch, num_workers=0)

    print(f"device: {device} | mode: {args.mode} | condition: {args.condition} "
          f"| WINDOW-level split (paper-style, leaky) | {n} windows\n")

    model = make_model(args.mode).to(device)

    labels = np.array([full.windows[i][1] for i in train_idx])
    counts = np.bincount(labels, minlength=6)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for accel, mic, y in train_dl:
            y = y.to(device)
            loss = criterion(forward(model, args.mode, accel, mic, device), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        yv, pv = predict(model, args.mode, val_dl, device)
        f1 = f1_score(yv, pv, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d} | val acc {accuracy_score(yv, pv):.3f} | val macro-F1 {f1:.3f}")

    model.load_state_dict(best_state)
    yt, pt = predict(model, args.mode, test_dl, device)
    print(f"\n===== {args.mode} / {args.condition}: WINDOW-level (paper-style) =====")
    print(f"accuracy : {accuracy_score(yt, pt):.3f}")
    print(f"macro-F1 : {f1_score(yt, pt, average='macro'):.3f}")
    print("confusion matrix (rows = true, cols = pred):")
    print("order:", CLASS_NAMES)
    print(confusion_matrix(yt, pt, labels=range(6)))


if __name__ == "__main__":
    main()
