import matplotlib.pyplot as plt

# Data from the spreadsheet
delta_s = [2500, 2000, 1500, 1250, 1000, 750, 500, 400, 300, 250, 200, 150, 125, 100, 75, 50]
avg_compile_time = [0.22, 0.24, 0.25, 0.28, 0.32, 0.44, 0.66, 0.77, 1.03, 1.05, 1.55, 2.67, 3.75, 3.50, 11.33, 35.34]
avg_total_energy = [186.7602891, 172.1498074, 158.6086560, 152.0182050, 145.4622289, 139.9948123, 132.6440147, 129.8018757, 127.1540419, 125.5741245, 124.1422694, 122.5420551, 121.7328884, 120.9573370, 119.9957765, 119.0255923]

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
ax1.invert_xaxis()
ax1.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.7)

ax2 = ax1.twinx()
ax2.set_ylabel('Average Compile Time', color=color2, fontsize=16, fontweight='bold')
ax2.plot(delta_s, avg_compile_time, marker='o', linestyle='-', color=color2, linewidth=3, markersize=9, label='Avg Compile Time')
ax2.tick_params(axis='y', labelcolor=color2)

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

ax2_log = ax1_log.twinx()
ax2_log.set_ylabel('Average Compile Time', color=color2, fontsize=16, fontweight='bold')
ax2_log.plot(delta_s, avg_compile_time, marker='o', linestyle='-', color=color2, linewidth=3, markersize=9, label='Avg Compile Time')
ax2_log.tick_params(axis='y', labelcolor=color2)

plt.title('Average Total Energy and Compile Time vs. delta_s (Logarithmic X)', fontsize=18, fontweight='bold')
fig_log.tight_layout(pad=2)
plt.show()