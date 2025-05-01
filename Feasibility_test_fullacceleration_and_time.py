# Convert speed from km/h to m/s: multiply by (1000/3600)
acceleration = 0.767  # m/s^2 (converted from km/h/s)
time_limit = 200  # seconds (5 minutes)
speed_cap_kmh = 160  # km/h
speed_cap = speed_cap_kmh * (1000/3600)  # convert to m/s

print("Time (s) | Speed (km/h) | Distance (m)")
print("------- | ----------- | --------")

current_speed = 0
distance = 0

for time in range(1, time_limit + 1):
    if current_speed < speed_cap:
        current_speed = min(acceleration * time, speed_cap)
    
    # Calculate distance (area under speed-time curve)
    if current_speed == speed_cap:
        # After reaching speed cap, distance increases linearly
        distance = (0.5 * speed_cap * speed_cap / acceleration) + (speed_cap * (time - speed_cap/acceleration))
    else:
        # Before reaching speed cap, use regular acceleration formula
        distance = 0.5 * acceleration * time**2
    
    # Convert current_speed to km/h for display
    current_speed_kmh = current_speed * (3600/1000)
    print(f"{time:7} | {current_speed_kmh:11.2f} | {distance:8.2f}")
