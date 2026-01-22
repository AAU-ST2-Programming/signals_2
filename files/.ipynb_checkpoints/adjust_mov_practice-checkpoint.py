# Moving average (for reference)
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, medfilt
import numpy as np
import matplotlib.pyplot as plt

# Juster window 
window = 1
filename = "b001.csv"



"""
Lad være med at ændre noget herunder. 
OGSÅ selvom i får en error.
(det gør i bl.a. hvis window er et lige tal)
"""

# HEY LOOK; Du kan også bare loade specifikke columns! (der er 5 columns i filen)
SCG, timestamp = np.loadtxt(filename,delimiter=",", skiprows=1, unpack=True, usecols=[3,4])

fc = 15  # cutoff Hz

x = SCG[:20000]
t = timestamp[:20000]
x_ma = uniform_filter1d(x,window)

# Butterworth zero-phase
fs = 5000 # sample rate Hz
b, a = butter(4, fc, fs=fs, btype='low')
x_bw = filtfilt(b, a, x)
# Median
x_med = medfilt(x, kernel_size=window)

plt.figure()
plt.plot(t, x, color='0.7', label='Rå data')
plt.plot(t, x_med, label='Median')
plt.plot(t, x_ma, label=f'MA (N={N})')
plt.plot(t, x_bw, label='Butterworth (filtfilt)')
plt.title('Udglatning af signalet')
plt.xlabel('Tid [s]')
plt.ylabel('Amplitude [a.u.]')
plt.legend()
plt.show()