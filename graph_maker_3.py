import numpy as np
import matplotlib.pyplot as plt

# Sensitivity table values
wacc = [10, 11, 12, 13, 14, 15]
g = [2.0, 2.5, 3.0, 3.5, 4.0]

values = np.array([
    [142.00, 147.86, 154.54, 162.26, 171.26],
    [123.66, 127.89, 132.65, 138.05, 144.21],
    [109.12, 112.27, 115.76, 119.66, 124.05],
    [97.35,  99.73,  102.36, 105.26, 108.48],
    [87.63,  89.47,  91.48,  93.69,  96.11],
    [79.49,  80.93,  82.50,  84.21,  86.07]
])

# NVIDIA-themed colors
nvidia_green = "#76B900"
dark_bg = "#0B0F0C"
text_color = "#E6F2E6"
grid_color = "#1E2A1E"

plt.figure(figsize=(10, 5.5), facecolor=dark_bg)
ax = plt.gca()
ax.set_facecolor(dark_bg)

# Heatmap
im = ax.imshow(values, aspect="auto")

# Labels
ax.set_xticks(range(len(g)))
ax.set_xticklabels([f"{x:.1f}%" for x in g], color=text_color)
ax.set_yticks(range(len(wacc)))
ax.set_yticklabels([f"{x}%" for x in wacc], color=text_color)

ax.set_xlabel("Terminal Growth Rate (g)", color=text_color)
ax.set_ylabel("WACC", color=text_color)
ax.set_title("Sensitivity Heatmap: Implied Share Price ($)", color=text_color, pad=14)

# Add value annotations
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        ax.text(j, i, f"${values[i, j]:.2f}", ha="center", va="center",
                color=text_color, fontsize=14)

# Grid lines
ax.set_xticks(np.arange(-0.5, len(g), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(wacc), 1), minor=True)
ax.grid(which="minor", color=grid_color, linestyle="-", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)

# Highlight base case cell (WACC=14%, g=3%)
base_i = wacc.index(14)
base_j = g.index(3.0)
ax.scatter(base_j, base_i, s=220, facecolors="none", edgecolors=nvidia_green, linewidths=2.5)

# Colorbar
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color=text_color)
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=text_color)
cbar.outline.set_edgecolor(grid_color)

plt.tight_layout()
plt.show()
