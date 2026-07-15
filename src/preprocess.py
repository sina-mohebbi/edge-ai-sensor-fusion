"""
preprocess.py  —  run once.
Convert the raw pump recordings (huge CSVs) into compact per-recording .npz files,
plus an index.csv holding the labels. We keep the FULL continuous signal so that
window length / overlap can be chosen later without re-running this.

Output (in data/processed/):
    <folder>.npz  ->  acc: float32 (n_acc, 3)   ax_g, ay_g, az_g   @ 4000 Hz
                      mic: int16   (n_mic, 2)    left, right        @ 48000 Hz
                      label_idx: int             0..5
    index.csv     ->  one row per recording: folder, label, condition, distance, lengths
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- paths (relative to this file, so it works from anywhere) ----
ROOT = Path(__file__).resolve().parent.parent      # project root
DATASET_DIR = ROOT / "Dataset"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
README = DATASET_DIR / "README.txt"

FS_ACC, FS_MIC = 4000, 48000
LABELS = ["nominal", "75", "50", "25", "20", "15"]  # severity order -> class 0..5


def parse_readme(path):
    """Walk the README top-to-bottom, remembering the current distance/label/condition,
    and attach them to each recording folder listed underneath."""
    records, distance, label, noisy = [], None, None, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "50 cm" in line:
            distance = 50; continue
        if "100 cm" in line:
            distance = 100; continue
        if line.replace("*", "").strip() == "":      # a pure ***** separator line
            continue
        if line.startswith("***"):                    # a label header, e.g. "*** Suction at 75% with PUMP 1 on"
            desc = line.lstrip("*").strip()
            noisy = "PUMP 1 on" in desc
            if "Nominal" in desc:
                label = "nominal"
            else:
                m = re.search(r"(\d+)\s*%", desc)
                label = m.group(1) if m else None
            continue
        m = re.match(r"(\d{8}_\d{6})\s*-\s*(\d+)s", line)   # a folder line
        if m:
            folder, dur = m.group(1), int(m.group(2))
            records.append(dict(folder=folder, label=label,
                                label_idx=LABELS.index(label),
                                condition="noisy" if noisy else "clean",
                                distance_cm=distance, dur_s=dur))
    return records


def load_acc(folder):
    p = DATASET_DIR / folder / "adxl355_4000Hz.csv"
    # columns 5,6,7 = ax_g, ay_g, az_g  (skip raw counts and the text 'type' column)
    return pd.read_csv(p, comment="#", header=None, usecols=[5, 6, 7],
                       dtype=np.float32).to_numpy()


def load_mic(folder):
    p = DATASET_DIR / folder / "mic_48000Hz.csv"
    # columns 2,3 = sample_left, sample_right
    arr = pd.read_csv(p, comment="#", header=None, usecols=[2, 3],
                      dtype=np.int32).to_numpy()
    assert np.abs(arr).max() < 32768, "mic sample exceeds int16 range"
    return arr.astype(np.int16)


def main():
    recs = parse_readme(README)
    print(f"README lists {len(recs)} recordings.")
    kept = []
    for r in tqdm(recs, desc="processing"):
        f = r["folder"]
        try:
            acc, mic = load_acc(f), load_mic(f)
        except Exception as e:
            print(f"  skip {f}: {e}"); continue
        if len(acc) < FS_ACC or len(mic) < FS_MIC:    # shorter than 1 second -> unusable
            print(f"  skip {f}: too short"); continue
        np.savez(OUT_DIR / f"{f}.npz", acc=acc, mic=mic, label_idx=r["label_idx"])
        r["n_acc"], r["n_mic"] = len(acc), len(mic)
        kept.append(r)

    idx = pd.DataFrame(kept)
    idx.to_csv(OUT_DIR / "index.csv", index=False)
    print(f"\nSaved {len(kept)} recordings to {OUT_DIR}")
    print("\nRecordings per label:\n", idx.groupby("label").size().to_string())
    print("\nRecordings per condition:\n", idx.groupby("condition").size().to_string())


if __name__ == "__main__":
    main()