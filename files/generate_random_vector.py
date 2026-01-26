import numpy as np

# ----------------------------
# Parameters
# ----------------------------
N = 1000                # total number of samples
mean1, std1 = 0.75, 0.025      # distribution 1


mean2, std2 = 0, 0.2      # distribution 2
p_dist1 = 0.5           # probability of choosing dist1

# ----------------------------
# Generate mixed data
# ----------------------------
# Generate time array
t = np.linspace(0, 4*np.pi, N)

# Generate sine wave around mean1 with noise
sine_signal = mean1 + 0.5 * np.sin(t)*std1*4 + np.random.normal(0, std1, N)

# Generate noise floor around mean2
noise_floor = np.random.normal(mean2, std2, N)

# Mix them based on probability
choice = np.random.rand(N) < p_dist1  # True = sine, False = noise floor
data = np.where(choice, sine_signal, noise_floor)


# ----------------------------
# Save to CSV
# ----------------------------
np.savetxt("random_dist.csv", data, delimiter=",")


import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,2)
axs[0].plot(data)
axs[0].set_xlabel("Sample")
axs[0].set_ylabel("Amplitude")
axs[0].set_title("Waveform")
axs[1].hist(data,   bins=50)
axs[1].set_xlabel("Amplitude")
axs[1].set_ylabel("Antal")
axs[1].set_title("Histogram")
plt.show()
