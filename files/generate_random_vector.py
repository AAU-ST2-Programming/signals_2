import numpy as np

# ----------------------------
# Parameters
# ----------------------------
N = 1000                # total number of samples
mean1, std1 = 0, 0.2      # distribution 1
mean2, std2 = 1, 0.2      # distribution 2
p_dist1 = 0.5           # probability of choosing dist1

# ----------------------------
# Generate mixed data
# ----------------------------
choice = np.random.rand(N) < p_dist1  # True = dist1, False = dist2
data = np.zeros(N)
data[choice] = np.random.normal(mean1, std1, np.sum(choice))
data[~choice] = np.random.normal(mean2, std2, np.sum(~choice))

# ----------------------------
# Save to CSV
# ----------------------------
np.savetxt("random_dist.csv", data, delimiter=",")


import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,1)
axs[0].plot(data)
axs[1].hist(data,   bins=50)
plt.show()
