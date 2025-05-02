import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, LogLocator

# Data from the spreadsheet (reversed for increasing delta_s)
delta_s = [50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 750, 1000, 1250, 1500, 2000, 2500]
avg_compile_time = [35.34, 11.33, 3.50, 3.75, 2.67, 1.55, 1.05, 1.03, 0.77, 0.66, 0.44, 0.32, 0.28, 0.25, 0.24, 0.22]
avg_total_energy = [119.0255923, 119.9957765, 120.9573370, 121.7328884, 122.5420551, 124.1422694, 125.5741245, 127.1540419, 129.8018757, 132.6440147, 139.9948123, 145.4622289, 152.0182050, 158.6086560, 172.1498074, 186.7602891]

# Set global font sizes and font family for academic style (slightly larger)
plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Computer Modern Roman', 'DejaVu Serif', 'serif']
})

# Create a combined plot with both on the same axes (with different scales)
fig, ax1 = plt.subplots(figsize=(14, 8))

color1 = 'tab:blue'
color2 = 'tab:red'

ax1.set_xlabel('delta_s', fontsize=16, fontweight='bold')
ax1.set_ylabel('Average Total Energy', color=color1, fontsize=16, fontweight='bold')
ax1.plot(delta_s, avg_total_energy, marker='o', linestyle='-', color=color1, linewidth=3, markersize=8, label='Avg Total Energy')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.7)

# Show x-axis from 0
ax1.set_xlim(left=0)

# Add more ticks to x and y axes
ax1.xaxis.set_major_locator(MultipleLocator(250))
ax1.xaxis.set_minor_locator(MultipleLocator(50))
ax1.yaxis.set_major_locator(MultipleLocator(10))
ax1.yaxis.set_minor_locator(MultipleLocator(2))

ax2 = ax1.twinx()
ax2.set_ylabel('Average Compile Time', color=color2, fontsize=16, fontweight='bold')
ax2.plot(delta_s, avg_compile_time, marker='o', linestyle='-', color=color2, linewidth=3, markersize=9, label='Avg Compile Time')
ax2.tick_params(axis='y', labelcolor=color2)

# Add more ticks to right y-axis
ax2.yaxis.set_major_locator(MultipleLocator(5))
ax2.yaxis.set_minor_locator(MultipleLocator(1))

plt.title('Average Total Energy and Compile Time vs. delta_s', fontsize=18, fontweight='bold')
fig.tight_layout(pad=2)
plt.show()

# New plot: Logarithmic x-axis for delta_s
fig_log, ax1_log = plt.subplots(figsize=(14, 8))

ax1_log.set_xlabel('delta_s (log scale)', fontsize=16, fontweight='bold')
ax1_log.set_ylabel('Average Total Energy', color=color1, fontsize=16, fontweight='bold')
ax1_log.plot(delta_s, avg_total_energy, marker='o', linestyle='-', color=color1, linewidth=3, markersize=8, label='Avg Total Energy')
ax1_log.tick_params(axis='y', labelcolor=color1)
ax1_log.set_xscale('log')
ax1_log.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.7)

# Add more ticks to log x-axis and y-axis
ax1_log.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax1_log.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
ax1_log.yaxis.set_major_locator(MultipleLocator(10))
ax1_log.yaxis.set_minor_locator(MultipleLocator(2))

ax2_log = ax1_log.twinx()
ax2_log.set_ylabel('Average Compile Time', color=color2, fontsize=16, fontweight='bold')
ax2_log.plot(delta_s, avg_compile_time, marker='o', linestyle='-', color=color2, linewidth=3, markersize=9, label='Avg Compile Time')
ax2_log.tick_params(axis='y', labelcolor=color2)

# Add more ticks to right y-axis
ax2_log.yaxis.set_major_locator(MultipleLocator(5))
ax2_log.yaxis.set_minor_locator(MultipleLocator(1))

plt.title('Average Total Energy and Compile Time vs. delta_s (Logarithmic X)', fontsize=18, fontweight='bold')
fig_log.tight_layout(pad=2)
plt.show()