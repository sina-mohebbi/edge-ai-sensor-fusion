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
    
class GatedFusionCNN(nn.Module):
    """Like FusionCNN, but instead of blindly concatenating the two branches, a
    small gate reads both feature vectors and learns how much to TRUST each
    modality (a weight in [0,1] each). The unreliable modality can be turned down
    so it can't drag the strong one along. Gates = 1 recovers plain concatenation,
    so this can only help."""

    def __init__(self, n_classes=6):
        super().__init__()
        self.accel_branch = ConvBranch(in_channels=1)
        self.mic_branch = ConvBranch(in_channels=2)
        d = self.accel_branch.out_dim

        self.gate = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * d, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, accel, mic):
        fa = self.accel_branch(accel)
        fm = self.mic_branch(mic)
        both = torch.cat([fa, fm], dim=1)
        g = torch.sigmoid(self.gate(both))
        fused = torch.cat([g[:, 0:1] * fa, g[:, 1:2] * fm], dim=1)
        return self.classifier(fused)


class SpecBranch(nn.Module):
    """A 2-D conv stack for spectrograms (freq x time), ending in a global
    average pool so any spectrogram size collapses to one feature vector."""

    def __init__(self, in_channels, widths=(16, 32, 64)):
        super().__init__()
        layers = [nn.BatchNorm2d(in_channels)]   # normalise the raw spectrogram
        c = in_channels
        for w in widths:
            layers += [
                nn.Conv2d(c, w, kernel_size=3, padding=1),
                nn.BatchNorm2d(w),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            c = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = c

    def forward(self, x):
        x = self.features(x)
        return self.pool(x).flatten(1)          # (batch, out_dim)


class SpectralFusionCNN(nn.Module):
    """The feature-rich model: turn each modality into a spectrogram (computed on
    the GPU inside forward), run a 2-D CNN per modality, then gate-fuse them.
    Uses ALL 3 accelerometer axes and both microphone channels — because
    cavitation is an impulsive, broadband, high-frequency phenomenon that shows up
    much more clearly in the time-frequency domain than in the raw waveform."""

    def __init__(self, n_classes=6, accel_channels=3):
        super().__init__()
        self.accel_branch = SpecBranch(accel_channels)   # 3 axes
        self.mic_branch = SpecBranch(2)                  # stereo
        d = self.accel_branch.out_dim + self.mic_branch.out_dim

        self.gate = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    @staticmethod
    def _spectrogram(x, n_fft, hop):
        # x: (batch, channels, length) -> log-magnitude STFT (batch, channels, freq, time)
        b, c, length = x.shape
        flat = x.reshape(b * c, length)
        window = torch.hann_window(n_fft, device=x.device)
        spec = torch.stft(flat, n_fft=n_fft, hop_length=hop,
                          window=window, return_complex=True).abs()
        spec = torch.log1p(spec)
        return spec.reshape(b, c, spec.shape[-2], spec.shape[-1])

    def forward(self, accel, mic):
        a_spec = self._spectrogram(accel, n_fft=256, hop=128)     # accel @ 4 kHz
        m_spec = self._spectrogram(mic, n_fft=1024, hop=512)      # mic  @ 48 kHz
        fa = self.accel_branch(a_spec)
        fm = self.mic_branch(m_spec)
        both = torch.cat([fa, fm], dim=1)
        g = torch.sigmoid(self.gate(both))
        fused = torch.cat([g[:, 0:1] * fa, g[:, 1:2] * fm], dim=1)
        return self.classifier(fused)


class HybridFeatureCNN(nn.Module):
    """The feature-rich model. For each modality it combines three views:
      (1) a raw 1-D CNN branch      -> time / amplitude cues,
      (2) a 2-D CNN branch on the spectrogram -> time-frequency cues,
      (3) a compact vector of hand-crafted descriptors per channel
          (log-RMS, kurtosis, crest factor, spectral centroid, HF ratio).
    Motivated by the physics: nominal/75/50 differ only by FLOW (amplitude, band
    energy), while 25/20/15 are cavitation (impulsive, high-frequency). The three
    views are concatenated and classified."""

    def __init__(self, n_classes=6, accel_channels=3, mic_channels=2):
        super().__init__()
        self.accel_raw = ConvBranch(accel_channels)     # 1-D, time domain
        self.mic_raw = ConvBranch(mic_channels)
        self.accel_spec = SpecBranch(accel_channels)    # 2-D, spectrogram
        self.mic_spec = SpecBranch(mic_channels)

        self.n_desc = 5                                 # descriptors per channel
        n_hand = self.n_desc * (accel_channels + mic_channels)
        self.hand_norm = nn.BatchNorm1d(n_hand)

        learned = (self.accel_raw.out_dim + self.mic_raw.out_dim
                   + self.accel_spec.out_dim + self.mic_spec.out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(learned + n_hand, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    @staticmethod
    def _stft_mag(x, n_fft, hop):
        b, c, length = x.shape
        flat = x.reshape(b * c, length)
        window = torch.hann_window(n_fft, device=x.device)
        mag = torch.stft(flat, n_fft=n_fft, hop_length=hop,
                        window=window, return_complex=True).abs()
        return mag.reshape(b, c, mag.shape[-2], mag.shape[-1])

    @staticmethod
    def _descriptors(x, mag):
        # x: (B,C,L) waveform ; mag: (B,C,F,T) linear-magnitude spectrogram
        eps = 1e-8
        xc = x - x.mean(dim=-1, keepdim=True)
        var = xc.pow(2).mean(dim=-1)
        rms = torch.sqrt(var + eps)                                 # amplitude (flow cue)
        crest = x.abs().amax(dim=-1) / (rms + eps)                  # impulsiveness
        kurt = xc.pow(4).mean(dim=-1) / (var.pow(2) + eps)          # impulsiveness
        n_freq = mag.shape[-2]
        freqs = torch.arange(n_freq, device=x.device).view(1, 1, n_freq, 1).float()
        centroid = (freqs * mag).sum(-2) / (mag.sum(-2) + eps)      # (B,C,T)
        centroid = centroid.mean(-1) / n_freq                       # (B,C), normalised
        hf_ratio = mag[:, :, n_freq // 2:, :].sum((-2, -1)) / (mag.sum((-2, -1)) + eps)
        feats = torch.stack([torch.log1p(rms), kurt, crest, centroid, hf_ratio], dim=-1)
        return feats.flatten(1)                                     # (B, C*5)

    def forward(self, accel, mic):
        a_mag = self._stft_mag(accel, n_fft=256, hop=128)
        m_mag = self._stft_mag(mic, n_fft=1024, hop=512)

        parts = [
            self.accel_raw(accel),
            self.mic_raw(mic),
            self.accel_spec(torch.log1p(a_mag)),
            self.mic_spec(torch.log1p(m_mag)),
            self.hand_norm(torch.cat(
                [self._descriptors(accel, a_mag), self._descriptors(mic, m_mag)], dim=1)),
        ]
        return self.classifier(torch.cat(parts, dim=1))


# quick shape/param check: python src/model.py
if __name__ == "__main__":
    raw12 = (torch.randn(4, 1, 4000), torch.randn(4, 2, 48000))
    raw32 = (torch.randn(4, 3, 4000), torch.randn(4, 2, 48000))
    for name, model, args in [
        ("fusion", FusionCNN(), raw12),
        ("gated", GatedFusionCNN(), raw12),
        ("spectral", SpectralFusionCNN(), raw32),
        ("hybrid", HybridFeatureCNN(), raw32),
    ]:
        out = model(*args)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:9s} output: {tuple(out.shape)}  params: {n_params:,}")