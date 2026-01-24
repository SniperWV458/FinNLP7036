import matplotlib.pyplot as plt

# Data
years = list(range(1, 11))
growth = [50, 35, 25, 15, 10, 10, 10, 10, 10, 10]

# NVIDIA-themed colors
nvidia_green = "#76B900"
dark_bg = "#0B0F0C"
text_color = "#E6F2E6"
grid_color = "#1E2A1E"
accent_gray = "#A7B1A7"

plt.figure(figsize=(10, 5.5), facecolor=dark_bg)
ax = plt.gca()
ax.set_facecolor(dark_bg)

# Plot
ax.plot(years, growth, marker="o", linewidth=2.5, markersize=7,
        color=nvidia_green, markerfacecolor=nvidia_green, markeredgecolor=text_color)

# Axes labels and title
ax.set_title("Revenue Growth Assumptions (Years 1–10)", color=text_color, pad=14)
ax.set_xlabel("Forecast Year", color=text_color)
ax.set_ylabel("Revenue Growth (%)", color=text_color)

# Ticks
ax.set_xticks(years)
ax.tick_params(axis="x", colors=text_color)
ax.tick_params(axis="y", colors=text_color)

# Grid
ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color=grid_color, alpha=0.9)
ax.xaxis.grid(False)

# Spines
for spine in ax.spines.values():
    spine.set_color(grid_color)

# Annotations (rationale callouts)
ax.annotate(
    "Near-term demand:\nH100/H200 backlog +\nhyperscaler AI CapEx",
    xy=(1, 50), xycoords="data",
    xytext=(2.2, 56), textcoords="data",
    arrowprops=dict(arrowstyle="->", color=accent_gray, lw=1.2),
    color=text_color, fontsize=10, ha="left", va="center"
)

ax.annotate(
    "Normalization:\nLaw of Large Numbers +\ncompetition (AMD / TPU / ASICs)",
    xy=(5, 10), xycoords="data",
    xytext=(6.2, 24), textcoords="data",
    arrowprops=dict(arrowstyle="->", color=accent_gray, lw=1.2),
    color=text_color, fontsize=10, ha="left", va="center"
)

# Highlight the normalization plateau
ax.axhline(10, color=accent_gray, linewidth=1.2, linestyle=":", alpha=0.9)
ax.text(10.05, 10, "10% long-run\n(Y5–Y10)", color=accent_gray, fontsize=9, va="center")

ax.set_ylim(0, 60)
plt.tight_layout()
plt.show()
