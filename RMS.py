import math
import matplotlib.pyplot as plt

def calculate_combined_journey_exact_time(speed_limit,target_velocity, max_acc, max_braking, m, C_d, A, C, eta, WindSpeed, v_init, max_p, braking_eff, total_time, delta_t, plot=False):
    """
    Combines the acceleration, cruising, and braking phases into a single journey,
    adjusting the cruising time to finish exactly at the given total time.
    """
    g = 9.807  # Gravitational acceleration (m/s²)
    rho_air = 1.225  # Air density (kg/m³)

    time = 0
    distance = 0
    total_energy = 0
    v = v_init
    acceleration = 0  # Initialize acceleration to zero

    distances = []
    velocities = []
    powers = []
    accelerations = []
    distances.append(0)
    velocities.append(0)
    powers.append(0)
    accelerations.append(0)

    # Acceleration phase
    while v < target_velocity:
        drag_force = 0.5 * rho_air * C_d * A * (v + WindSpeed)**2
        rolling_resistance = C * m * g
        acceleration = (max_p*eta/(v+1e-6) - drag_force - rolling_resistance) / m
        if acceleration > max_acc:
            acceleration = max_acc
            power = v * (drag_force + rolling_resistance + m * acceleration)/eta  # Power calculation
            # print(f"if acceleartion>max_acc: {power/1000} W, in speed: {v*3.6} m/s")
        else:
            if v * (drag_force + rolling_resistance + m * acceleration)/eta > max_p:
                acceleration = (max_p*eta/(v+1e-6) - drag_force - rolling_resistance) / m
                power = max_p
                # print(f"otherwise: {power} W")
            else:
                power = v * (drag_force + rolling_resistance + m * acceleration)/eta  # Power calculation
        
        v_new = v + acceleration * delta_t        
        if v_new <= target_velocity:
            avg_velocity = (v + v_new) / 2
            v = v_new
        else:
            avg_velocity = (v + target_velocity) / 2
            v = target_velocity

        time += delta_t
        distance += avg_velocity * delta_t
        total_energy += power * delta_t / 3.6e6

        distances.append(distance / 1000)
        velocities.append(v * 3.6)
        powers.append(power / 1e6)
        accelerations.append(acceleration)


    # Estimate remaining time for cruising and braking
    braking_time = target_velocity / max_braking
    cruising_time = total_time - time - braking_time
    
    # Cruising phase
    if cruising_time > 0:
        for _ in range(int(cruising_time / delta_t)):  # Divide cruise time into 0.1s intervals
            drag_force = 0.5 * rho_air * C_d * A * (v + WindSpeed)**2
            rolling_resistance = C * m * g

            
            avg_velocity = v
            time += delta_t
            distance += avg_velocity * delta_t

            power = avg_velocity / eta * (drag_force + rolling_resistance)
            # power = min(power, max_p)
            total_energy += power * delta_t / 3.6e6

            distances.append(distance / 1000)
            velocities.append(v * 3.6)
            powers.append(power / 1e6)
            accelerations.append(0)  # No acceleration during cruising

    # Braking phase
    while v > 0:
        drag_force = 0.5 * rho_air * C_d * A * (v + WindSpeed)**2
        rolling_resistance = C * m * g
        deceleration = max_braking

        v_new = max(0, v - deceleration * delta_t)
        avg_velocity = (v + v_new) / 2
        v = v_new
        time += delta_t
        distance += avg_velocity * delta_t

        power = avg_velocity * (drag_force + rolling_resistance - m * deceleration)
        regenerative_power = power * braking_eff if power < 0 else 0
        total_energy += (regenerative_power) * delta_t / 3.6e6

        distances.append(distance / 1000)
        velocities.append(v * 3.6)
        powers.append((regenerative_power) / 1e6)
        accelerations.append(-deceleration)

    # Convert distance to kilometers
    distance_km = distance / 1000

    # print(f"Total time for journey: {time:.2f} seconds")")
    # print(f"Total distance for journey: {distance_km:.3f} km")")
    if plot == True:
        print(f"Total energy consumption: {total_energy:.3f} kWh")
        plot_velocity_and_power_combined(distances, velocities, powers, accelerations)
        save_velocity_and_power_data("RMS_results.csv", distances, velocities, powers, accelerations, total_energy=total_energy)

    # Return the total distance traveled
    if plot:
        return distance_km, total_energy
    return distance_km

def plot_velocity_and_power_combined(distances, velocities, powers, accelerations):
    """
    Plots velocity, power consumption, and acceleration profiles on the same plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot velocity on the primary y-axis
    ax1.plot(distances, velocities, 'r-', label='Velocity (km/h)', linewidth=2)
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Velocity (km/h)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    # Plot power on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(distances, powers, 'b-', label='Power (MW)', linewidth=2)
    ax2.set_ylabel('Power (MW)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.grid(visible=True, which='major', axis='y', linestyle='--', alpha=0.7)  # Major grid lines
    ax2.grid(visible=True, which='minor', axis='y', linestyle=':', alpha=0.5)  # Minor grid lines
    ax2.minorticks_on()  # Enable minor ticks for power y-axis

    # Plot acceleration on a third y-axis
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.plot(distances, accelerations, 'g-', label='Acceleration (m/s²)', linewidth=2)
    ax3.set_ylabel('Acceleration (m/s²)', color='g')
    ax3.tick_params(axis='y', labelcolor='g')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

    plt.title('Velocity, Power Consumption, and Acceleration vs Distance')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    # plt.show()

def save_velocity_and_power_data(filepath, distances, velocities, powers, accelerations, total_energy=None):
    """
    Saves velocity, power, and acceleration data to a file.
    Optionally appends total energy consumption at the end.
    """
    with open(filepath, 'w') as file:
        file.write("Distance (km),Velocity (km/h),Power (MW),Acceleration (m/s²)\n")
        for d, v, p, a in zip(distances, velocities, powers, accelerations):
            file.write(f"{d:.3f},{v:.3f},{p:.3f},{a:.3f}\n")
        if total_energy is not None:
            file.write("\nTotal Energy Consumption (kWh):{:.3f}\n".format(total_energy))

if __name__ == "__main__":
    max_target_velocity = 44  # Target velocity in m/s (160 km/h)
    speed_limit = 44
    max_acc = 0.768  # Maximum acceleration in m/s²
    max_braking = 0.5  # Maximum braking in m/s²
    
    m = 391000 * (1 + 0.06)  # Train mass in kg
    C_d = 0.8  # Drag coefficient
    A = 3.02*4.67  # Frontal area in m²
    C = 0.002  # Rolling resistance coefficient
    WindSpeed = 0  # Wind speed in m/s
    v_init = 0  # Initial velocity in m/s
    max_p = 359900*6  # Maximum power in watts (from Train.py)
    
    braking_eff = 0.9  # Regenerative braking efficiency (80%)
    eta = 1  # Efficiency
    total_time = 380  # Total time for the journey in seconds
    total_distance = 10000  # Total distance for the journey in m
    difference = 100000
    delta_t = 1  # Time step for calculations in seconds
    

    target_velocity_final = 0  # Initialize with a default value
    closest_difference = float('inf')  # Initialize with a large value

    
    for target_velocity in [i * 0.1 for i in range(1, int(max_target_velocity * 10) + 1)]:  # Increment by 0.1 m/s
        # print(f"Calculating for target velocity: {target_velocity} m/s")
        distance = calculate_combined_journey_exact_time(
            speed_limit,target_velocity, max_acc, max_braking, m, C_d, A, C, eta, WindSpeed, v_init, max_p, braking_eff, total_time, delta_t, plot=False
        )
        difference = abs(total_distance - distance * 1000)  # Convert distance to meters for comparison
        

        if difference < closest_difference:
            closest_difference = difference
            closest_distance = distance * 1000
            target_velocity_final = target_velocity

    print(f"Optimal target velocity for {total_time} seconds: {target_velocity_final*3.6:.2f} km/h")
    print(f"Closest distance: {closest_distance / 1000:.3f} km (Target distance: {total_distance / 1000:.3f} km)")
    
    # Calculate and plot the results for the optimal target velocity
if __name__ == "__main__":
    # ...existing code...
    # Calculate and plot the results for the optimal target velocity
    distances, velocities, powers = [], [], []
    _, total_energy = calculate_combined_journey_exact_time(
        speed_limit, target_velocity_final, max_acc, max_braking, m, C_d, A, C, eta, WindSpeed, v_init, max_p, braking_eff, total_time, delta_t, plot=True
    )
    print("Results saved to 'RMS_results.csv'.")

