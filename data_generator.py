# Example Python code to generate speed_limits.txt and power_limits.txt
total_time = 600  # Total time in seconds
total_distance = 10000  # Total distance in meters
delta_s = 100
# Generate gradients
Gradients = [0.001 for i in range(total_distance//delta_s//4)] + \
               [0.001 for i in range(total_distance//delta_s//4, total_distance//delta_s//2)] + \
               [0.001 for i in range(total_distance//delta_s//2, total_distance//delta_s*3//4)] + \
               [0.001 for i in range(total_distance//delta_s*3//4, total_distance//delta_s)]

# Write to gradients.txt
with open('gradients.txt', 'w') as f:
    for gradient in Gradients:
        f.write(f"{gradient}\n")

# Generate speed limits
speed_limits = [44 for i in range(total_time//4)] + \
               [44 for i in range(total_time//4, total_time//2)] + \
               [44 for i in range(total_time//2, total_time*3//4)] + \
               [44 for i in range(total_time*3//4, total_time)]

# Write to speed_limits.txt
with open('speed_limits.txt', 'w') as f:
    for speed in speed_limits:
        f.write(f"{speed}\n")

# Generate power limits
power_limits_1 = [2.5 * 10**6 for i in range(total_time//4)] + \
               [2.5 * 10**6 for i in range(total_time//4, total_time//2)] + \
               [2.5 * 10**6 for i in range(total_time//2, total_time*3//4)] + \
               [2.5 * 10**6 for i in range(total_time*3//4, total_time)]

# Write to power_limits.txt
with open('power_limits_1.txt', 'w') as f:
    for power in power_limits_1:
        f.write(f"{power}\n")

# Generate power limits
power_limits_2 = [2.5 * 10**6 for i in range(total_time//4)] + \
               [2.5 * 10**6 for i in range(total_time//4, total_time//2)] + \
               [2.5 * 10**6 for i in range(total_time//2, total_time*3//4)] + \
               [2.5 * 10**6 for i in range(total_time*3//4, total_time)]

# Write to power_limits.txt
with open('power_limits_2.txt', 'w') as f:
    for power in power_limits_2:
        f.write(f"{power}\n")