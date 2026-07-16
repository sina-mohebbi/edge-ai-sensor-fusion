# dataset.py
# A PyTorch Dataset that serves 1-second windows of (accelerometer, microphone).
#
# It reads split.csv to know which recordings belong to train / val / test, then
# builds a flat list of windows across those recordings. Windows are cut on the
# fly in __getitem__ so we don't store overlapping copies on disk.
#
# The accel and mic streams have different rates (4 kHz vs 48 kHz) but start at
# the same instant, so a window that starts at accel-sample s starts at mic-sample
# 12*s (since 48000/4000 = 12). That's how the two stay time-aligned.

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent.parent
PROCESSED = HERE / "data" / "processed"

FS_ACC, FS_MIC = 4000, 48000
RATIO = FS_MIC // FS_ACC            # 12
ACCEL_AXIS = 2                      # 0=x 1=y 2=z — paper uses one "vertical" channel; z carries the clearest trend
MIC_SCALE = 32768.0                # int16 range -> roughly [-1, 1]


class CavitationWindows(Dataset):
    def __init__(self, split, window_sec=1.0, overlap=0.5):
        # which recordings are in this split, with their labels
        index = pd.read_csv(PROCESSED / "index.csv")
        assign = pd.read_csv(PROCESSED / "split.csv")
        recs = index.merge(assign, on="folder")
        recs = recs[recs["split"] == split].reset_index(drop=True)

        self.win_acc = int(window_sec * FS_ACC)
        self.win_mic = int(window_sec * FS_MIC)
        hop_acc = max(1, int(self.win_acc * (1 - overlap)))
        hop_mic = hop_acc * RATIO

        # flat list of windows: (folder, label, accel_start, mic_start)
        self.windows = []
        for _, r in recs.iterrows():
            n_acc, n_mic = int(r["n_acc"]), int(r["n_mic"])
            n_from_acc = 1 + (n_acc - self.win_acc) // hop_acc
            n_from_mic = 1 + (n_mic - self.win_mic) // hop_mic
            n = min(n_from_acc, n_from_mic)          # stay inside both streams
            for w in range(n):
                self.windows.append(
                    (r["folder"], int(r["label_idx"]), w * hop_acc, w * hop_mic)
                )

        self._cache = {}   # folder -> (acc array, mic array), loaded once

    def __len__(self):
        return len(self.windows)

    def _recording(self, folder):
        if folder not in self._cache:
            data = np.load(PROCESSED / f"{folder}.npz")
            self._cache[folder] = (data["acc"], data["mic"])
        return self._cache[folder]

    def __getitem__(self, i):
        folder, label, s_acc, s_mic = self.windows[i]
        acc, mic = self._recording(folder)

        a = acc[s_acc:s_acc + self.win_acc, ACCEL_AXIS].astype(np.float32)
        m = mic[s_mic:s_mic + self.win_mic, :].astype(np.float32)

        # remove DC / gravity (keeps the amplitude differences that matter),
        # and scale the mic to a sane range
        a = a - a.mean()
        m = (m - m.mean(axis=0)) / MIC_SCALE

        # conv1d wants (channels, length)
        accel = torch.from_numpy(a).unsqueeze(0)      # (1, 4000)
        micro = torch.from_numpy(m.T).contiguous()    # (2, 48000)
        return accel, micro, label


# quick self-test: python src/dataset.py
if __name__ == "__main__":
    for name in ["train", "val", "test"]:
        ds = CavitationWindows(name, overlap=0.5 if name == "train" else 0.0)
        a, m, y = ds[0]
        print(f"{name:5s}: {len(ds):5d} windows | accel {tuple(a.shape)} "
              f"mic {tuple(m.shape)} label {y}")