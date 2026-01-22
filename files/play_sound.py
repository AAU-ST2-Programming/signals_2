import sounddevice as sd
from scipy.signal import resample
import numpy as np

filename = "files/b001.csv" # Hvis du får en FileNotFoundError, prøv at fjern "files/" fra string.

data = np.loadtxt(filename,delimiter=",", skiprows=1, unpack=True, usecols=[3])
fs = 5000 # orginal sampling frekvens
fs_new = 44100 # sounddevive kan ikke finde ud af at resample til noget brugbart selv, så det skal du selv gøre!
num_samples = int(len(data) * fs_new / fs)
data_resampled = resample(data, num_samples)
sd.play(data_resampled, fs_new)
sd.wait()           # Wait until playback finishes