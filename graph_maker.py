import matplotlib.pyplot as plt

# ===== Inputs (your numbers) =====
re = 14.03                  # cost of equity (%)
rd_pre_tax = 5.26           # cost of debt (%)
tax = 14.68 / 100           # corporate tax rate
we = 99.78 / 100            # equity weight
wd = 0.22 / 100             # debt weight

# ===== Calculations =====
rd_after_tax = rd_pre_tax * (1 - tax)
equity_contrib = we * re
debt_contrib = wd * rd_after_tax
wacc = equity_contrib + debt_contrib

# ===== NVIDIA-themed colors =====
nvidia_green = "#76B900"
dark_bg = "#0B0F0C"
text_color = "#E6F2E6"
grid_color = "#1E2A1E"
accent_gray = "#A7B1A7"

labels = ["Equity contribution\n(we × Re)", "Debt contribution\n(wd × Rd × (1−T))"]
values = [equity_contrib, debt_contrib]
colors = [nvidia_green, "#3A4A3A"]  # green + muted dark green/gray

plt.figure(figsize=(9, 5.2), facecolor=dark_bg)
ax = plt.gca()
ax.set_facecolor(dark_bg)

bars = ax.bar(labels, values, color=colors, edgecolor=nvidia_green, linewidth=1.5)

ax.set_ylabel("Contribution to WACC (%)", color=text_color)
ax.set_title(f"WACC Breakdown (NVDA) — Total WACC = {wacc:.2f}%", color=text_color, pad=14)

ax.set_ylim(0, max(values) * 1.25)

# Grid styling
ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color=grid_color, alpha=0.8)
ax.xaxis.grid(False)

# Ticks styling
ax.tick_params(axis="x", colors=text_color)
ax.tick_params(axis="y", colors=text_color)

# Spine styling
for spine in ax.spines.values():
    spine.set_color(grid_color)

# Annotate bars
for b in bars:
    y = b.get_height()
    ax.text(
        b.get_x() + b.get_width() / 2,
        y + 0.05,
        f"{y:.2f}%",
        ha="center",
        va="bottom",
        color=text_color,
        fontsize=17,
        fontweight="bold"
    )

plt.tight_layout()
plt.show()

print(f"After-tax Rd = {rd_after_tax:.2f}%")
print(f"WACC = {wacc:.2f}%")
