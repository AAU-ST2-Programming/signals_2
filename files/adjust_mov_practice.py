# Moving average (for reference)
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, medfilt
import numpy as np
import matplotlib.pyplot as plt

# Juster window 
window = 1
filename = "files/b001.csv" # Hvis du får en FileNotFoundError, prøv at fjern "files/" fra string.



"""
Lad være med at ændre noget herunder. 
OGSÅ selvom i får en error.
(det gør i bl.a. hvis window er et lige tal)
"""

# HEY LOOK; Du kan også bare loade specifikke columns! (der er 5 columns i filen)
SCG, timestamp = np.loadtxt(filename,delimiter=",", skiprows=1, unpack=True, usecols=[3,4])

fc = 15  # cutoff Hz

x = SCG[:20000] # 4 sekunder af SCG signalet
t = timestamp[:20000] 

# moving average filter (du kan lave et MOV på MANGE måder)
x_ma = uniform_filter1d(x,window)

# Butterworth zero-phase
fs = 5000 # sample rate Hz
b, a = butter(4, fc, fs=fs, btype='low') # type: ignore
x_bw = filtfilt(b, a, x)
# Median
x_med = medfilt(x, kernel_size=window)

plt.figure()
plt.plot(t, x, color='0.7', label='Rå data')
plt.plot(t, x_med, label='Median')
plt.plot(t, x_bw, label='Butterworth (filtfilt)')
plt.plot(t, x_ma, label=f'MOV ({window=})')
plt.title('Udglatning af signalet')
plt.xlabel('Tid [s]')
plt.ylabel('Amplitude [a.u.]')
plt.legend()


import sounddevice as sd


sd.play(SCG, 44100)

sd.wait()           # Wait until playback finishes
plt.show()