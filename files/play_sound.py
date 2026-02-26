import sounddevice as sd
from scipy.signal import resample
import numpy as np

filename = "files/b001.csv"  # Hvis du får en FileNotFoundError, prøv at fjern "files/" fra string.
data = np.loadtxt(filename, delimiter=",", skiprows=1, unpack=True, usecols=[3])
fs = 5000  # orginal sampling frekvens
sd.play(data, fs)
sd.wait()  # Wait until playback finishes
