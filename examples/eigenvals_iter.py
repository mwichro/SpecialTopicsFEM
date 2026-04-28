import numpy as np
import matplotlib.pyplot as plt

N = 32 # Number of internal nodes
x = np.linspace(0, 1, N+2)[1:-1] # Physical grid points

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot the 3 LOWEST frequency eigenvectors
for m in [1, 2, 3]:
    v_m = np.sin(m * np.pi * x)
    ax1.plot(x, v_m, label=f'm={m} (Low Freq)', marker='.')

ax1.set_title("Low Frequency Error Modes (Stalls)")
ax1.legend()

# Plot the 3 HIGHEST frequency eigenvectors
for m in [N-2, N-1, N]:
    v_m = np.sin(m * np.pi * x)
    ax2.plot(x, v_m, label=f'm={m} (High Freq)', marker='.')

ax2.set_title("High Frequency Error Modes")
ax2.legend()
plt.show()