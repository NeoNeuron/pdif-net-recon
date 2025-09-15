#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Load Lorenz attractor data from files
x = np.fromfile('./tmp/Lp=0.00s=2.000f=0.000u=0.000_x.dat', dtype=float).reshape(-1, 2)[:,1]
y = np.fromfile('./tmp/Lp=0.00s=2.000f=0.000u=0.000_y.dat', dtype=float).reshape(-1, 2)[:,1]
z = np.fromfile('./tmp/Lp=0.00s=2.000f=0.000u=0.000_z.dat', dtype=float).reshape(-1, 2)[:,1]

# Create 3D plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
ax.plot(x, y, z, lw=0.5, color='#3532A0')
ax.axis('off')
# Set the face color of the three axis planes to white
plt.savefig('./tmp/lorenz_attractor.png', dpi=300, transparent=True)  # Save with transparent background

# %%
# Load Lorenz attractor data from files
x = np.fromfile('./tmp/Rconp=0.00s=0.002_x.dat', dtype=float).reshape(-1, 2)[:,1]
y = np.fromfile('./tmp/Rconp=0.00s=0.002_y.dat', dtype=float).reshape(-1, 2)[:,1]
z = np.fromfile('./tmp/Rconp=0.00s=0.002_z.dat', dtype=float).reshape(-1, 2)[:,1]

# Create 3D plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
ax.plot(x, y, z, lw=0.5, color='#3532A0')
ax.axis('off')
# Set the face color of the three axis planes to white
plt.savefig('./tmp/Rossler_attractor.png', dpi=300, transparent=True)  # Save with transparent background

#%%
# Load Logistic map data from files
x = np.fromfile('./tmp/Logp=0.00s=0.005_voltage.dat', dtype=float).reshape(-1, 2)[:,1]
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)

x_grid = np.linspace(0.0, 1.0, 400)
ax.plot(x_grid, 4*x_grid*(1-x_grid), lw=2.5, c='grey', clip_on=False)
ax.plot([0,1], [0,1], lw=2.5, c='grey', ls='--')
# Plot trajectory
for i in range(4):
    ax.plot([x[i], x[i+1], x[i+1]], [x[i+1], x[i+1], x[i+2]], 'o', ms=10, color='#3532A0', clip_on=False)
    ax.annotate(
        '', 
        xy=(x[i+1], x[i+2]), 
        xytext=(x[i+1], x[i+1]), 
        arrowprops=dict(arrowstyle='-|>', color='#3532A0', lw=1, mutation_scale=40)  # Change mutation_scale to adjust arrow size
    )
    ax.annotate(
        '', 
        xy=(x[i+1], x[i+1]), 
        xytext=(x[i], x[i+1]), 
        arrowprops=dict(arrowstyle='-|>', color='#3532A0', lw=1, mutation_scale=40)  # Change mutation_scale to adjust arrow size
    )
# ax.axis('off')
ax.axis('scaled')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(labelsize=20)
# Set the face color of the three axis planes to white
plt.savefig('./tmp/Logistic_map.png', dpi=300, transparent=True)  # Save with transparent background