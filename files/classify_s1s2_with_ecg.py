# %load files/classify_s1s2_with_ecg.py
#!/usr/bin/env python3
"""
classify_s1s2_with_ecg.py

Detect S1 and S2 in PCG using EKG R-peaks as temporal anchors.

Rules implemented (brief):
1) Simple amplitude gating: keep peaks above a fraction of the envelope std to
   avoid classifying noise as heart sounds.
2) S1 is expected shortly after the R-peak (approx 10-150 ms). We search for a local
   maximum of the PCG envelope in this window.
3) S2 is expected later (approx 250-600 ms after R). We search for a local maximum
   in that window and require it to be reasonably sized relative to the local envelope.

Produces: files/peaks_ECGPCG2.csv and a PNG with plots in the same folder.
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
def envelope(x, window=50):
    """
    Beregn en envelope for et signal x.
    """
    # 1. Fjern gennemsnittet
    mean_x = np.mean(x)
    s0 = x - mean_x

    # 2. Rectify (absolut værdi)
    r = np.abs(s0)

    # 3. Moving Maximum
    mov_max = np.zeros_like(r)
    half_w = window // 2
    for i in range(len(r)):
        start = max(i - half_w, 0)
        end = min(i + half_w + 1, len(r))
        mov_max[i] = np.max(r[start:end])

    # 4. Tilføj gennemsnittet tilbage
    env = mov_max + mean_x
    return env


class CardiacCycle:
    def __init__(self, r_idx, fs):
        self.r_idx = r_idx
        self.s1_idx = np.nan
        self.s1_amp = np.nan
        self.s2_idx = np.nan
        self.s2_amp = np.nan
        self.fs = fs

    def r_to_s1(self):
        return (self.s1_idx - self.r_idx) / self.fs
    def r_to_s2(self):
        return (self.s2_idx - self.r_idx) / self.fs
    def s1_to_s2(self):
        return (self.s2_idx - self.s1_idx) / self.fs
    
    @property
    def R(self):
        return (self.r_idx / self.fs) if (self.r_idx is not np.nan) else np.nan
    
    @property
    def S1(self):
        return (self.s1_idx / self.fs) if (self.s1_idx is not np.nan) else np.nan
    @property
    def S2(self):
        return (self.s2_idx / self.fs) if (self.s2_idx is not np.nan) else np.nan
    
    def __repr__(self):
        return f"CardiacCycle(R={self.R}, S1={self.S1}, S2={self.S2})"
    
    def as_list(self):
        return [self.R, self.S1, self.S2]


def bandpass(x, fs, low, high, order=3):
    b, a = butter(order, [low, high],fs=fs, btype='bandpass')
    return filtfilt(b, a, x)


def main():
    # use relative paths to the signals_2/files folder (run from repo root)
    ecgpcg_path = 'files/ECGPCG.csv'
    peaks_path = 'files/peaks_ECGPCG.csv'
    fig_path = 'files/s1s2_classification.png'
    out_path = 'files/peaks_ECGPCG2.csv'

    # read ECG/PCG using numpy (timestamp, ECG, PCG)
    ts, ecg, pcg = np.loadtxt(ecgpcg_path, delimiter=',', skiprows=1, unpack=True)
    # sampling frequency (robust median diff)
    dt = np.median(np.diff(ts))
    fs = 1.0 / dt

    # bandpass filter PCG, and compute envelope
    pcg = bandpass(pcg, fs=fs, low=20, high=300)
    pcg_env = envelope(pcg, window=101)
    
    # load peaks using numpy: classification (str), timestamp (float), amplitude (float)
    cls_col, index_col, amp_col = np.loadtxt(peaks_path, delimiter=',', dtype=str, skiprows=1, unpack=True)
    index_col = index_col.astype(float)
    amp_col = amp_col.astype(float)
    S = 0
    cardiac_cycles: list[CardiacCycle] = []
    C = None
    s1_win = (0.01, 0.15)   # 25-150 ms after R
    s2_win = (0.25, 0.60)    # 250-450 ms after R
    for i in range(len(cls_col)):   
        if cls_col[i] == "R":
            if C is not None:
                cardiac_cycles.append(C)
            C = CardiacCycle(r_idx=index_col[i], fs=fs)
            continue

        elif cls_col[i] == "PCG_PEAK":
            if C is None:
                continue
            S = index_col[i]

            # Implement Rules for S1 and S2 classification
            if s1_win[0] < (S - C.r_idx) / C.fs < s1_win[1] and np.isnan(C.s1_idx):
                # S1 rule
                C.s1_idx = index_col[i]
                C.s1_amp = amp_col[i]
            elif s2_win[0] < (S - C.r_idx) / C.fs < s2_win[1] and np.isnan(C.s2_idx):
                # S2 rule
                C.s2_idx = index_col[i]
                C.s2_amp = amp_col[i]
    if C is not None:
        cardiac_cycles.append(C)    


    # Visualize results
    s1_times = [c.S1 for c in cardiac_cycles]
    s1_amps = [c.s1_amp for c in cardiac_cycles]
    s2_times = [c.S2 for c in cardiac_cycles]
    s2_amps = [c.s2_amp for c in cardiac_cycles]

    r_to_s1 = [c.r_to_s1() for c in cardiac_cycles if not np.isnan(c.s1_idx)]
    r_to_s2 = [c.r_to_s2() for c in cardiac_cycles if not np.isnan(c.s2_idx)]
    s1_to_s2 = [c.s1_to_s2() for c in cardiac_cycles if not np.isnan(c.s1_idx) and not np.isnan(c.s2_idx)]

    # Plot results
    _,axs = plt.subplots(3,1,figsize=(12, 8))
    axs[0].plot(ts, ecg, label='ECG', color='k')
    axs[1].plot(ts, pcg, label='PCG', color='gray')
    axs[1].plot(ts, pcg_env, label='PCG Envelope', color='g')
    for c in cardiac_cycles:
        axs[0].axvline(c.R, color='r', linestyle='--', alpha=0.5)   
        axs[1].axvline(c.R, color='r', linestyle='--', alpha=0.5)   
    axs[1].scatter(s1_times, s1_amps, label='S1', color='b', marker='o')
    axs[1].scatter(s2_times, s2_amps, label='S2', color='r', marker='x')
    axs[0].set_title('ECG with R-peaks')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_xlim(ts[0], ts[-1])
    axs[0].legend()
    axs[1].set_title('PCG Signal')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_xlim(ts[0], ts[-1])
    axs[1].set_title('PCG Envelope with S1 and S2')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_xlim(ts[0], ts[-1])
    axs[1].legend()


    axs[2].hist(r_to_s1, bins=7, alpha=0.7, label='R to S1', color='b')
    axs[2].hist(r_to_s2, bins=7, alpha=0.7, label='R to S2', color='r')
    axs[2].hist(s1_to_s2, bins=7, alpha=0.7, label='S1 to S2', color='g')
    axs[2].set_title('Histogram of R to S1 and S2 Intervals') 
    axs[2].set_xlabel(r'$\Delta$ Time (s) from R peak')
    axs[2].set_ylabel('Count')
    axs[2].legend() 
    plt.tight_layout()
    

    plt.savefig(fig_path)
    
    # save results to CSV using np — replace None with NaN and ensure numeric dtype
    list_with_results = np.array([c.as_list() for c in cardiac_cycles], dtype=float)
    header = 'R,S1,S2'
    fmt = '%.3f'
    np.savetxt(out_path, list_with_results, delimiter=',', header=header, fmt=fmt)

    
    # This code locks, so we save the figure after saving to data of 
    plt.show()


if __name__ == '__main__':
    main()

