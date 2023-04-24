import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
cmap = np.array([i[:] for i in mcolors.TABLEAU_COLORS])
X = np.loadtxt('umap')
theta, phi = np.loadtxt('thetaphi')
print(theta.shape, phi.shape)
print(np.round(phi, decimals=3))
idx_cmap = np.int8(np.round(phi, decimals=3))
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection='3d')
plt.xlim(0, 15)
plt.ylim(-5, 12)
ax.scatter(*X.T, c=phi, linewidths=.5)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection='3d')
plt.xlim(0, 15)
plt.ylim(-5, 12)
ax.scatter(*X.T, c=theta, linewidths=.5)
plt.show()