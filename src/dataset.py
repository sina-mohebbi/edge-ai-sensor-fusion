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
ACCEL_AXIS = 2                      # 0=x 1=y 2=z; z shows the clearest trend for the 1-D models
MIC_SCALE = 32768.0                # int16 range -> roughly [-1, 1]


class CavitationWindows(Dataset):
    def __init__(self, split=None, folders=None, window_sec=1.0, overlap=0.5,
                 accel_axes=(ACCEL_AXIS,), augment=False):
        # accel_axes selects which accelerometer channels to return:
        #   (2,)       -> just the z axis (the 1-D models)
        #   (0, 1, 2)  -> all three axes (the spectral model)
        self.accel_axes = list(accel_axes)

        # augment adds small random changes to each window (training only).
        # It uses numpy's global random, so seeding numpy makes it reproducible.
        self.augment = augment

        # Pick the recordings for this dataset. Either pass a split name
        # ("train"/"val"/"test", read from split.csv) or an explicit list of
        # folders (used by cross-validation, which builds its own folds).
        index = pd.read_csv(PROCESSED / "index.csv")
        if folders is not None:
            recs = index[index["folder"].isin(folders)].reset_index(drop=True)
        else:
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

        a = acc[s_acc:s_acc + self.win_acc][:, self.accel_axes].astype(np.float32)
        m = mic[s_mic:s_mic + self.win_mic, :].astype(np.float32)

        # remove DC / gravity per channel (keeps the amplitude differences that
        # matter), and scale the mic to a sane range
        a = a - a.mean(axis=0)
        m = (m - m.mean(axis=0)) / MIC_SCALE

        # during training, nudge the window a little so the model sees more variety
        if self.augment:
            time_slide = np.random.uniform(-0.1, 0.1)     # same slide for both signals
            volume = 1.0 + np.random.uniform(-0.08, 0.08)
            a = self._vary(a, time_slide, volume)
            m = self._vary(m, time_slide, volume)

        # conv layers want (channels, length)
        accel = torch.from_numpy(a.T).contiguous()    # (n_axes, 4000)
        micro = torch.from_numpy(m.T).contiguous()    # (2, 48000)
        return accel, micro, label

    def _vary(self, signal, time_slide, volume):
        # slide the window in time, change the volume a touch, and add faint noise
        n = len(signal)
        signal = np.roll(signal, int(time_slide * n), axis=0)
        signal = signal * volume
        noise = np.random.standard_normal(signal.shape).astype(np.float32)
        signal = signal + noise * (0.02 * signal.std())
        return signal.astype(np.float32)


# quick self-test: python src/dataset.py
if __name__ == "__main__":
    for name in ["train", "val", "test"]:
        ds = CavitationWindows(name, overlap=0.5 if name == "train" else 0.0)
        a, m, y = ds[0]
        print(f"{name:5s}: {len(ds):5d} windows | accel {tuple(a.shape)} "
              f"mic {tuple(m.shape)} label {y}")