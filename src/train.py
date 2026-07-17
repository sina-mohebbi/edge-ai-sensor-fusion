# train.py
# Trains a model on one train/val/test split and reports test metrics.
#
#   python src/train.py --mode fusion   # sound + vibration
#   python src/train.py --mode accel    # vibration only
#   python src/train.py --mode mic      # sound only
#
# Adam, lr 1e-3, batch 32, with class weighting, best model kept on val macro F1,
# and a scheduler that drops the learning rate on a plateau.

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dataset import CavitationWindows
from model import FusionCNN, SingleModalityCNN, GatedFusionCNN

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
CLASS_NAMES = ["nominal", "75", "50", "25", "20", "15"]


def class_weights(ds, device):
    # inverse-frequency weights so rare classes (20%, 15%) aren't ignored
    labels = np.array([w[1] for w in ds.windows])
    counts = np.bincount(labels, minlength=6)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def forward(model, mode, accel, mic, device):
    accel, mic = accel.to(device), mic.to(device)
    if mode in ("fusion", "gated"):
        return model(accel, mic)
    return model(accel if mode == "accel" else mic)


@torch.no_grad()
def evaluate(model, mode, loader, device):
    model.eval()
    trues, preds = [], []
    for accel, mic, y in loader:
        out = forward(model, mode, accel, mic, device)
        preds.append(out.argmax(1).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fusion", "gated", "accel", "mic"], default="fusion")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | mode: {args.mode}")

    train_ds = CavitationWindows("train", overlap=0.5, augment=True)
    val_ds = CavitationWindows("val", overlap=0.0)
    test_ds = CavitationWindows("test", overlap=0.0)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.batch, num_workers=0)

    if args.mode == "fusion":
        model = FusionCNN()
    elif args.mode == "gated":
        model = GatedFusionCNN()
    else:
        model = SingleModalityCNN(1 if args.mode == "accel" else 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen, running = 0, 0.0
        for accel, mic, y in train_dl:
            y = y.to(device)
            out = forward(model, args.mode, accel, mic, device)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            seen += len(y)

        yv, pv = evaluate(model, args.mode, val_dl, device)
        val_acc = accuracy_score(yv, pv)
        val_f1 = f1_score(yv, pv, average="macro")
        scheduler.step(val_f1)
        print(f"epoch {epoch:2d} | loss {running/seen:.3f} | "
              f"val acc {val_acc:.3f} | val macro-F1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # final test with the best checkpoint
    model.load_state_dict(best_state)
    yt, pt = evaluate(model, args.mode, test_dl, device)
    print("\n===== TEST =====")
    print(f"accuracy : {accuracy_score(yt, pt):.3f}")
    print(f"macro-F1 : {f1_score(yt, pt, average='macro'):.3f}")
    print("confusion matrix (rows = true, cols = pred):")
    print("order:", CLASS_NAMES)
    print(confusion_matrix(yt, pt, labels=range(6)))

    torch.save(best_state, RESULTS / f"model_{args.mode}.pt")
    print(f"\nsaved -> results/model_{args.mode}.pt")


if __name__ == "__main__":
    main()
