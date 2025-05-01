import numpy as np
def check_feasibility(total_time):
    # Calculate theoretical bounds
    S = 10000
    max_v = 44.444 # m/s = 160 km/h (VIRM)
    max_acc = 0.768 # m/s2 (2.76 km/h/s) (VIRM)
    avg_speed_required = S / total_time  # m/s
    max_possible_speed = min(max_v, np.sqrt(2 * max_acc * S))  # m/s
    min_possible_time = 2 * np.sqrt(S / max_acc)  # Theoretical minimum time
    
    print(f"Time: {total_time}s")
    print(f"Required average speed: {avg_speed_required*3.6:.2f} km/h")
    print(f"Maximum possible speed: {max_possible_speed*3.6:.2f} km/h")
    print(f"Theoretical minimum time: {min_possible_time:.2f}s")
    print(f"Feasible? {avg_speed_required <= max_possible_speed and total_time >= min_possible_time}")

# Check both cases
check_feasibility(386)
check_feasibility(387)