import pickle
import numpy as np
from scipy import signal
import os
import pandas as pd
import json

default_frequency_bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100),
}


def compute_band_power(data, fs=30_000, frequency_bands=None):
    if frequency_bands is None:
        frequency_bands = default_frequency_bands

    targets = []

    # compute spectrogram across all channels
    for channel_idx in range(data.shape[1]):
        channel_data = data[:, channel_idx]

        nperseg = int(fs * 0.5)  # 0.5 second windows
        freqs, psd = signal.welch(channel_data, fs, nperseg=nperseg)

        row = {}
        for band_name, (low, high) in frequency_bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.mean(psd[mask])
            if psd[mask].size == 0:
                band_power = 0.0

            # peak_freq = freqs[mask][np.argmax(psd[mask])]

            row[band_name] = band_power

        targets.append(row)

    return pd.DataFrame(targets)


if __name__ == "__main__":

    with open(os.path.expandvars("$SCRATCH/data/miv-recording_0000.pkl"), "rb") as f:
        recording = pickle.load(f)

    data = recording["data"]  # (n_samples, n_channels)
    timestamps = recording["timestamps"]  # (n_samples,)
    fs = recording["rate"]

    print(f"Samples: {data.shape}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Recording duration: {len(timestamps)/fs} seconds")

    q = compute_band_power(data, fs=fs)

    stats = {
        "mean": q.mean().to_dict(),
        "std": q.std().to_dict(),
        "bands": default_frequency_bands,
        "channels": q.to_dict(),
        "fs": fs,
    }

    with open("band_power_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
