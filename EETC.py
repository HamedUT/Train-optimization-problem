import matplotlib.pyplot as plt

def calculate_combined_journey_exact_time(total_distance, delta_t, speed_limit, max_acc, max_braking, m, C_d, A, C, eta, WindSpeed, v_init, max_p, braking_eff, total_time, plot=False):
    """
    Combines the acceleration, cruising, and braking phases into a single journey,
    adjusting the cruising time to finish exactly at the given total time.
    """
    closest_difference = float('inf')  # Initialize with a large value
    g = 9.807  # Gravitational acceleration (m/s²)
    rho_air = 1.225  # Air density (kg/m³)
    for braking_speed in [i * 0.1 for i in range(0, 10*int(speed_limit) + 1)]:  # Increment by 0.1 m/s
        time = 0
        distance = 0
        total_energy = 0
        v = v_init
        acceleration = 0  # Initialize acceleration to zero
        distances = []
        velocities = []
        powers = []
        accelerations = []
        times = []  # Add time tracking
        distances.append(0)
        velocities.append(0)
        powers.append(0)
        accelerations.append(0)
        times.append(0)  # Initial time

        # Acceleration phase
        while v < speed_limit:
            drag_force = 0.5 * rho_air * C_d * A * (v + WindSpeed)**2
            rolling_resistance = C * m * g
            acceleration = (max_p*eta/(v+1e-6) - drag_force - rolling_resistance) / m
            if acceleration > max_acc:
                acceleration = max_acc
                power = v * (drag_force + rolling_resistance + m * acceleration)/eta  # Power calculation
            else:
                if v * (drag_force + rolling_resistance + m * acceleration)/eta > max_p:
                    acceleration = (max_p*eta/(v+1e-6) - drag_force - rolling_resistance) / m
                    power = max_p
                else:
                    power = v * (drag_force + rolling_resistance + m * acceleration)/eta  # Power calculation
            
            v_new = v + acceleration * delta_t        
            if v_new <= speed_limit:
                avg_velocity = (v + v_new) / 2
                v = v_new
            else:
                avg_velocity = (v + speed_limit) / 2
                v = speed_limit

            time += delta_t
            distance += avg_velocity * delta_t
            total_energy += power * delta_t / 3.6e6

            distances.append(distance / 1000)
            velocities.append(v * 3.6)
            powers.append(power / 1e6)
            accelerations.append(acceleration)
            times.append(time)  # Track time
        acceleration_time = time
        # Estimate remaining time for cruising, coasting, and braking
        braking_time = braking_speed / max_braking
        coasting_plus_cruising_time = total_time - time - braking_time

        if coasting_plus_cruising_time > 0:
            coasting_time = 0
            cruising_time_counter = 0
            distance_coasting = 0
            # Saving Coasting phase with different lists. The reason is that first I ahve to know how long does it take to coast,
            # because I know how long it takes to cruise. In the next step, cruising is added to the original lists, 
            # because after acceleration, cruising happens and then coasting. after the original lists from cruising, 
            # the coasting lists are added.
            distances_coast = []
            velocities_coast = []
            powers_coast = []
            accelerations_coast = []
            times_coast = []  # Track time during coasting
            
            # Coasting phase 
            while v > braking_speed:
                drag_force = 0.5 * rho_air * C_d * A * (v + WindSpeed)**2
                rolling_resistance = C * m * g
                deceleration = (drag_force + rolling_resistance)/m
                v_new = v - deceleration * delta_t
                avg_velocity = (v + v_new) / 2
                v = v_new
                coasting_time += delta_t
                distance_coasting += avg_velocity * delta_t
                power = 0
                distances_coast.append(distance_coasting / 1000)
                velocities_coast.append(v * 3.6)
                powers_coast.append(power / 1e6)
                accelerations_coast.append(deceleration)
                times_coast.append(acceleration_time + cruising_time_counter + coasting_time)  # Track time during coasting
                
            cruising_time = coasting_plus_cruising_time - coasting_time
            if cruising_time < 0:
                continue
            time += coasting_time
            # Cruising phase 
            v = speed_limit
            if cruising_time > 0:    
                for _ in range(int(cruising_time / delta_t)):
                    drag_force = 0.5 * rho_air * C_d * A * (v + WindSpeed)**2
                    rolling_resistance = C * m * g                
                    avg_velocity = v
                    cruising_time_counter += delta_t
                    distance += avg_velocity * delta_t
                    power = avg_velocity / eta * (drag_force + rolling_resistance)
                    total_energy += power * delta_t / 3.6e6
                    distances.append(distance / 1000)
                    velocities.append(v * 3.6)
                    powers.append(power / 1e6)
                    accelerations.append(0)  # No acceleration during cruising
                    times.append(acceleration_time + cruising_time_counter)  # Track time during cruising
                time += cruising_time_counter
            distances.extend([d + distance/1000 for d in distances_coast])
            velocities.extend(velocities_coast)
            powers.extend(powers_coast)
            accelerations.extend(accelerations_coast)
            times.extend(times_coast)  # Add coasting times to the main list
            distance += distance_coasting
            v = braking_speed

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
            times.append(time)  # Track time during braking
         
        difference = abs(total_distance - distance)  # Convert distance to meters for comparison
        if difference < closest_difference:
            closest_difference = difference
            braking_speed_final = braking_speed
            total_energy_final = total_energy*1.064
            distance_final = distance 
            if plot == True:
                distances_final = distances
                velocities_final = velocities
                powers_final = powers
                accelerations_final = accelerations
                times_final = times  # Store final times
                acceleration_time_final = acceleration_time
                cruising_time_final = cruising_time
                coasting_time_final = coasting_time
                braking_time_final = braking_time
                  
    if plot == True:
        print(f"Difference: {closest_difference:.3f} m, braking: {braking_speed_final*3.6:.1f} km/h, Max speed: {speed_limit*3.6:.1f} km/h, acc time: {acceleration_time_final:.1f} cruising: {cruising_time_final:.1f} s, coasting: {coasting_time_final:.1f} s, braking time: {braking_time_final:.1f} s")
        print(f"Total energy consumption: {total_energy_final:.3f} kWh, braking speed: {braking_speed_final*3.6:.1f} km/h")
        plot_velocity_and_power_combined(distances_final, velocities_final, powers_final, accelerations_final)
        save_velocity_and_power_data("EETC_results.csv", distances_final, velocities_final, powers_final, accelerations_final, times_final, total_energy=total_energy_final)
        # Return the total distance traveled
    return braking_speed_final, distance_final 

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

def save_velocity_and_power_data(filepath, distances, velocities, powers, accelerations, times, total_energy=None):
    """
    Saves velocity, power, acceleration, and time data to a file.
    Optionally appends total energy consumption at the end.
    """
    with open(filepath, 'w') as file:
        file.write("Distance (km),Time (s),Velocity (km/h),Power (MW),Acceleration (m/s²)\n")
        for d, t, v, p, a in zip(distances, times, velocities, powers, accelerations):
            file.write(f"{d:.3f},{t:.2f},{v:.3f},{p:.3f},{a:.3f}\n")
        if total_energy is not None:
            file.write("\nTotal Energy Consumption (kWh):,{:.3f}\n".format(total_energy))

if __name__ == "__main__":
    speed_limit_allowed = 44
    delta_t = 1  # Time step in seconds
    max_acc = 0.768  # Maximum acceleration in m/s²
    max_braking = 0.5  # Maximum braking in m/s²
    
    m = 391000  # Train mass in kg
    C_d = 0.8  # Drag coefficient
    A = 3.02*4.67  # Frontal area in m²
    C = 0.002  # Rolling resistance coefficient
    WindSpeed = 0  # Wind speed in m/s
    v_init = 0  # Initial velocity in m/s
    max_p = 359900*6  # Maximum power in watts (from Train.py)
    
    braking_eff = 0.9  # Regenerative braking efficiency (80%)
    eta = 1  # Efficiency
    total_time = 360  # Total time for the journey in seconds
    total_distance = 10000  # Total distance for the journey in m
    difference = 100000

closest_difference = float('inf')  # Initialize with a large value
for speed_limit in [i * 0.2 for i in range(0, 5*int(speed_limit_allowed) + 1)]:  # Increment by 0.1 m/s
    braking_speed_final, distance_final = calculate_combined_journey_exact_time(total_distance, delta_t, speed_limit, max_acc, max_braking, m, C_d, A, C, eta, WindSpeed, v_init, max_p, braking_eff, total_time, plot=False)    
    difference = abs(total_distance - distance_final)  # Convert distance to meters for comparison
    if difference < closest_difference:            
        closest_difference = difference
        speed_limit_final = speed_limit

calculate_combined_journey_exact_time(total_distance, delta_t, speed_limit_final, max_acc, max_braking, m, C_d, A, C, eta, WindSpeed, v_init, max_p, braking_eff, total_time, plot=True)    
# calculate_combined_journey_exact_time(total_distance, delta_t, 44, max_acc, max_braking, m, C_d, A, C, eta, WindSpeed, v_init, max_p, braking_eff, total_time, plot=True)