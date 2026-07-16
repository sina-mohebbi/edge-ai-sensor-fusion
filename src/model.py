# model.py
# The intermediate-fusion CNN from the paper: two parallel 1-D conv branches
# (one for vibration, one for sound), whose features are concatenated and sent
# to a small classifier. Also here: a single-branch version, so we can reproduce
# the paper's accel-only and mic-only baselines with the same building blocks.

import torch
import torch.nn as nn


class ConvBranch(nn.Module):
    """A stack of conv blocks with a growing number of filters, ending in a
    global average pool so any input length collapses to one feature vector."""

    def __init__(self, in_channels, widths=(16, 32, 64, 128)):
        super().__init__()
        layers = []
        c = in_channels
        for w in widths:
            layers += [
                nn.Conv1d(c, w, kernel_size=7, padding=3),
                nn.BatchNorm1d(w),
                nn.ReLU(),
                nn.MaxPool1d(4),          # shrink the time axis 4x each block
            ]
            c = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = c

    def forward(self, x):
        x = self.features(x)
        return self.pool(x).squeeze(-1)   # (batch, out_dim)


class FusionCNN(nn.Module):
    """Vibration branch + sound branch -> concatenate features -> classify."""

    def __init__(self, n_classes=6):
        super().__init__()
        self.accel_branch = ConvBranch(in_channels=1)   # vertical acceleration
        self.mic_branch = ConvBranch(in_channels=2)     # stereo audio
        fused_dim = self.accel_branch.out_dim + self.mic_branch.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, accel, mic):
        fa = self.accel_branch(accel)
        fm = self.mic_branch(mic)
        fused = torch.cat([fa, fm], dim=1)
        return self.classifier(fused)


class SingleModalityCNN(nn.Module):
    """One branch only — used for the accel-only and mic-only baselines."""

    def __init__(self, in_channels, n_classes=6):
        super().__init__()
        self.branch = ConvBranch(in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.branch.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.branch(x))


# quick shape/param check: python src/model.py
if __name__ == "__main__":
    model = FusionCNN()
    accel = torch.randn(4, 1, 4000)     # a dummy batch of 4 windows
    mic = torch.randn(4, 2, 48000)
    out = model(accel, mic)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"fusion output: {tuple(out.shape)}  (expect (4, 6))")
    print(f"parameters: {n_params:,}")