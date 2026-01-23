# %load files/adjust_mov_practice.py
# Moving average (for reference)
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, medfilt
import numpy as np
import matplotlib.pyplot as plt

# Juster window 
window = 1
filename = "files/ECGPCG.csv" # Hvis du får en FileNotFoundError, prøv at fjern "files/" fra string.



"""
Lad være med at ændre noget herunder. 
OGSÅ selvom i får en error.
(det gør i bl.a. hvis window er et lige tal)
"""
fs = 8000 # sample rate Hz

# HEY LOOK; Du kan også bare loade specifikke columns! (der er 5 columns i filen)
timestamp, ecg, pcg = np.loadtxt(filename,delimiter=",", skiprows=1, unpack=True)


x = pcg[:fs*4] # 4 sekunder af SCG signalet
t = timestamp[:fs*4] 

# moving average filter (du kan lave et MOV på MANGE måder)
x_ma = uniform_filter1d(x,window)


plt.figure()
plt.plot(t, x, color='0.7', label='Rå data')
plt.plot(t, x_ma, label=f'MOV ({window=})')
plt.title('Udglatning af signalet')
plt.xlabel('Tid [s]')
plt.ylabel('Amplitude [a.u.]')
plt.xlim([1.6, 2.3])
plt.legend()
plt.show()