#!/usr/bin/env python3
"""
Simple script to classify S1/S2 using EKG R-peaks as reference.
Saves results to `files/peaks_ECGPCG.csv` with columns: classification,timestamp,amplitude

Run: python files/classify_s1s2_with_ecg.py
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt

# Sampling frequency in the dataset
fs = 8000


def envelope(x, window=51):
    """Beregn envelope via glidende maksimum af rectificeret signal."""
    mean_x = np.mean(x)
    s0 = x - mean_x
    r = np.abs(s0)
    mov_max = np.zeros_like(r)
    half_w = window // 2
    for i in range(len(r)):
        start = max(i - half_w, 0)
        end = min(i + half_w + 1, len(r))
        mov_max[i] = np.max(r[start:end])
    return mov_max + mean_x


def bandpass(x, low, high, fs, order=4):
    b, a = butter(order, [low, high], fs=fs, btype='bandpass')
    return filtfilt(b, a, x)


def main():
    filename = 'files/ECGPCG.csv'

    # Indlæs data
    timestamp, ecg, pcg = np.loadtxt(filename, delimiter=',', skiprows=1, unpack=True)

    # Forbered EKG: let filtrering + envelope
    ecg_f = bandpass(ecg, 1, 300, fs)
    env_ecg = envelope(ecg_f, window=41)

    # Find R-peaks i EKG-envelopen
    r_idxs, _ = find_peaks(env_ecg, height=np.std(env_ecg)*0.8, distance=int(0.25*fs))

    # Forbered PCG: bandpass + envelope
    pcg_f = bandpass(pcg, 50, 300, fs)
    env_pcg = envelope(pcg_f, window=301)
    pcg_idxs, _ = find_peaks(env_pcg, height=np.std(env_pcg)*0.8, distance=int(0.12*fs))

    # Klassifikation vha. R-reference: S1 i [60,200] ms efter R
    s1_window_ms = (60, 200)
    s1_samples = (int(s1_window_ms[0] / 1000.0 * fs), int(s1_window_ms[1] / 1000.0 * fs))

    labels = []  # tuples (classification, timestamp, amplitude)
    used_pcg = set()

    for r in r_idxs:
        s = r + s1_samples[0]
        e = min(r + s1_samples[1], len(env_pcg) - 1)
        if s >= e:
            continue
        # find PCG peaks in window
        candidates = [p for p in pcg_idxs if (p >= s and p <= e and p not in used_pcg)]
        if len(candidates) > 0:
            s1 = candidates[0]
            labels.append(('S1', float(timestamp[s1]), float(env_pcg[s1])))
            used_pcg.add(s1)
            # next unused PCG peak after s1 -> S2
            next_peaks = [p for p in pcg_idxs if (p > s1 and p not in used_pcg)]
            if len(next_peaks) > 0:
                s2 = next_peaks[0]
                labels.append(('S2', float(timestamp[s2]), float(env_pcg[s2])))
                used_pcg.add(s2)

    # Tilføj R-peaks (valgfrit, men nyttigt) til output
    for r in r_idxs:
        labels.append(('R', float(timestamp[r]), float(env_ecg[r])))

    # Sortér efter tid og gem
    labels_sorted = sorted(labels, key=lambda x: x[1])
    df = pd.DataFrame(labels_sorted, columns=['classification', 'timestamp', 'amplitude'])
    outpath = 'files/peaks_ECGPCG.csv'
    df.to_csv(outpath, index=False)
    print(f"Saved {len(df)} labels to {outpath}")

    # Enkel validering: beregn R->S1 forsinkelse (ms) og print μ±σ
    r2s1 = []
    # map timestamps to indices for quick lookup
    ts_to_idx = {float(timestamp[i]): i for i in range(len(timestamp))}
    # find S1 entries and match to nearest previous R
    s1_rows = df[df['classification'] == 'S1']
    r_rows = df[df['classification'] == 'R']
    # convert to numpy arrays of times
    r_times = r_rows['timestamp'].to_numpy() if not r_rows.empty else np.array([])
    s1_times = s1_rows['timestamp'].to_numpy() if not s1_rows.empty else np.array([])
    for s1t in s1_times:
        # find nearest R before s1t
        candidates = r_times[r_times < s1t]
        if candidates.size == 0:
            continue
        rtime = candidates[-1]
        r2s1.append((s1t - rtime) * 1000.0)

    if len(r2s1) > 0:
        mu = np.mean(r2s1)
        sd = np.std(r2s1, ddof=1) if len(r2s1) > 1 else 0.0
        print(f"R->S1: μ = {mu:.1f} ms, ±1σ = {sd:.1f} ms (n={len(r2s1)})")


if __name__ == '__main__':
    main()
