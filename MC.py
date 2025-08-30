import matplotlib.pyplot as plt
import os
import csv

def calculate_combined_journey_exact_time(total_distance, delta_t, speed_limit, max_acc, max_braking, m, C_d, A, C, eta, WindSpeed, v_init, max_p, braking_eff, total_time, plot=False):
    """
    Combines the acceleration, cruising, and braking phases into a single journey,
    adjusting the cruising time to finish exactly at the given total time.
    """
    best_combined_error = float('inf')
    braking_speed_final = 0
    distance_final = 0
    total_time_final = 0
    total_energy_final=0
    acceleration_time_final = 0
    coasting_time_final = 0
    braking_time_final = 0

    distances_final,velocities_final, powers_final = [], [], []
    # Add time tracking
    times_final = []
    

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
        times = []
        distances.append(0)
        velocities.append(0)
        powers.append(0)
        accelerations.append(0)
        times.append(0)
        prev_timestep_power = 0
        prev_timestep_regenerative_power = 0

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
            avg_power = (power +prev_timestep_power)/2
            prev_timestep_power = power

            time += delta_t
            distance += avg_velocity * delta_t
            total_energy += avg_power * delta_t / 3.6e6

            distances.append(distance / 1000)
            velocities.append(v * 3.6)
            powers.append(power / 1e6)
            accelerations.append(acceleration)
            times.append(time)
        acceleration_time = time
        # Estimate remaining time for cruising, coasting, and braking
        braking_time = braking_speed / max_braking
        coasting_time = total_time - time - braking_time

        if coasting_time < 0:
            continue
        coasting_time = 0
        
        # Coasting phase 
        while v > braking_speed:
            drag_force = 0.5 * rho_air * C_d * A * (v + WindSpeed)**2
            rolling_resistance = C * m * g
            deceleration = (drag_force + rolling_resistance)/m
            v_new = v - deceleration * delta_t
            avg_velocity = (v + v_new) / 2
            v = v_new
            coasting_time += delta_t
            time += delta_t
            distance += avg_velocity * delta_t
            power = 0
            distances.append(distance / 1000)
            velocities.append(v * 3.6)
            powers.append(power / 1e6)
            accelerations.append(deceleration)
            times.append(time)

        v = braking_speed

        # Braking phase    
        while v > 0:
            drag_force = 0.5 * rho_air * C_d * A * (v + WindSpeed)**2
            rolling_resistance = C * m * g
            v_new = max(0, v - max_braking * delta_t)
            avg_velocity = (v + v_new) / 2
            v = v_new
            time += delta_t
            distance += avg_velocity * delta_t
            power = avg_velocity * (drag_force + rolling_resistance - m * max_braking)
            regenerative_power = power * braking_eff if power < 0 else 0
            avg_regenerative_power = (regenerative_power +prev_timestep_regenerative_power)/2
            prev_timestep_regenerative_power = regenerative_power
            total_energy += (avg_regenerative_power) * delta_t / 3.6e6
            distances.append(distance / 1000)
            velocities.append(v * 3.6)
            powers.append((regenerative_power) / 1e6)
            accelerations.append(-deceleration)
            times.append(time)
        
        total_time_temp = acceleration_time + coasting_time + braking_time
        time_difference = abs(total_time_temp - total_time)
        difference = abs(total_distance - distance)
        combined_error = 2*time_difference / total_time + difference / total_distance  # normalized sum
        

        if combined_error < best_combined_error:
            closest_difference = difference
            best_combined_error = combined_error
            braking_speed_final = braking_speed
            total_energy_final = total_energy*1.025
            distance_final = distance 
            acceleration_time_final = acceleration_time
            coasting_time_final = coasting_time
            braking_time_final = braking_time
            total_time_final = acceleration_time_final + coasting_time_final + braking_time_final
            
            distances_final = distances
            velocities_final = velocities
            powers_final = powers
            accelerations_final = accelerations
            times_final = times
                

    if plot == True:
        print(f"Final Dist: {distance_final:.3f} m, Total time: {total_time_final:.1f} s, braking: {braking_speed_final*3.6:.1f} km/h, Max speed: {speed_limit*3.6:.1f} km/h, acc time: {acceleration_time_final:.1f}, coast: {coasting_time_final:.1f} s, braking: {braking_time_final:.1f} s")
        print(f"Total energy consumption: {total_energy_final:.3f} kWh, braking speed: {braking_speed_final*3.6:.1f} km/h")
        plot_velocity_and_power_combined(distances_final, velocities_final, powers_final, accelerations_final)
        plot_velocity_and_power_vs_time(times_final, velocities_final, powers_final, accelerations_final)  # New plot
        save_velocity_and_power_data("MC_results.csv", distances_final, velocities_final, powers_final, accelerations_final, times_final, total_energy=total_energy_final)
        # Return the total distance traveled
    return distances_final,velocities_final, powers_final, braking_speed_final, distance_final, total_time_final, times_final, total_energy_final, speed_limit, acceleration_time_final,coasting_time_final, braking_time_final

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

def plot_velocity_and_power_vs_time(times, velocities, powers, accelerations):
    """
    Plots velocity, power consumption, and acceleration profiles vs time.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot velocity on the primary y-axis
    ax1.plot(times, velocities, 'r-', label='Velocity (km/h)', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (km/h)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    # Plot power on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(times, powers, 'b-', label='Power (MW)', linewidth=2)
    ax2.set_ylabel('Power (MW)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.grid(visible=True, which='major', axis='y', linestyle='--', alpha=0.7)
    ax2.grid(visible=True, which='minor', axis='y', linestyle=':', alpha=0.5)
    ax2.minorticks_on()

    # Plot acceleration on a third y-axis
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.plot(times, accelerations, 'g-', label='Acceleration (m/s²)', linewidth=2)
    ax3.set_ylabel('Acceleration (m/s²)', color='g')
    ax3.tick_params(axis='y', labelcolor='g')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

    plt.title('Velocity, Power Consumption, and Acceleration vs Time')
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
            file.write("\nTotal Energy Consumption (kWh):{:.3f}\n".format(total_energy))

if __name__ == "__main__":
    speed_limit_allowed = 44
    delta_t = 1  # Time step in seconds
    C_d = 0.8  # Drag coefficient
    A = 3.02*4.67  # Frontal area in m²
    C = 0.002  # Rolling resistance coefficient
    WindSpeed = 0  # Wind speed in m/s
    v_init = 0  # Initial velocity in m/s
    difference = 10000000
    # delta_s = 25 # should be the same with Train.py if i want the output be the input of initializer

    for max_acc in [0.5,0.7,0.9]:
        for max_braking in [0.3,0.5,0.7]:
            for m in [390000]:
                for max_p in [1500000,2000000,2500000]:
                    for braking_eff in [0.7,0.8,0.9]:
                        for eta in [0.85,0.9,0.95]:
                            for total_time in [360,420,480]:
                                for total_distance in [8000,10000,12000]:


                                    closest_difference = float('inf')  # Initialize with a large value
                                    for speed_limit in [i * 0.5 for i in range(0, 2*int(speed_limit_allowed) + 1)]:  # Increment by 0.1 m/s
                                        # Unpack times_final as well
                                        distances,velocities, powers, braking_speed_final, distance_final, total_time_final, times_final, _,_,_,_,_ = calculate_combined_journey_exact_time(total_distance, delta_t, speed_limit, max_acc, max_braking, m, C_d, A, C, eta, WindSpeed, v_init, max_p, braking_eff, total_time, plot=False)    
                                        difference = abs(total_distance - distance_final)  # Convert distance to meters for comparison
                                        if difference < closest_difference:            
                                            closest_difference = difference
                                            speed_limit_final = speed_limit
                                            distances_final,velocities_final, powers_final, times_final_final = distances,velocities, powers, times_final
                                    if closest_difference < 100:
                                        _,_, _, braking_speed_final_, distance_final_, total_time_final_, times_final_, total_energy_final, speed_limit, acceleration_time_final,coasting_time_final, braking_time_final = calculate_combined_journey_exact_time(total_distance, delta_t, speed_limit_final, max_acc, max_braking, m, C_d, A, C, eta, WindSpeed, v_init, max_p, braking_eff, total_time, plot=True)    
                                    else:
                                        _,_, _, braking_speed_final_, distance_final_, total_time_final_, times_final_, total_energy_final, speed_limit, acceleration_time_final,coasting_time_final, braking_time_final = calculate_combined_journey_exact_time(total_distance, delta_t, speed_limit_final, max_acc, max_braking, m, C_d, A, C, eta, WindSpeed, v_init, max_p, braking_eff, total_time, plot=False)    
                                    # Append scenario result to CSV
                                    output_file = "MC_scenarios.csv"
                                    fieldnames = [
                                        "max_acc", "max_braking", "m", "max_p", "eta", "braking_eff", "total_time","total_distance",
                                        "distance_final_", "total_time_final_", "total_energy_final",
                                        "acceleration_time_final", "coasting_time_final", "braking_time_final",
                                        "braking_speed_final_","eligible"]
                                    if closest_difference < 100 and abs(total_time_final_ - total_time) < 10:
                                        eligible = 1 
                                    else: 
                                        eligible = 0
                                    file_exists = os.path.exists(output_file)
                                    with open(output_file, "a", newline="") as f:
                                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                                        if not file_exists or os.stat(output_file).st_size == 0:
                                            writer.writeheader()
                                        writer.writerow({
                                            "max_acc": max_acc,
                                            "max_braking": max_braking,
                                            "m": m,
                                            "max_p": max_p,
                                            "eta": eta,
                                            "braking_eff": braking_eff,
                                            "total_time": total_time,
                                            "total_distance": total_distance,
                                            "distance_final_": distance_final_,
                                            "total_time_final_": total_time_final_,
                                            "total_energy_final": total_energy_final,
                                            "acceleration_time_final": acceleration_time_final,
                                            "coasting_time_final": coasting_time_final,
                                            "braking_time_final": braking_time_final,
                                            "braking_speed_final_": braking_speed_final_*3.6,
                                            "eligible": eligible
                                        })
                                    




# def linear_interpolate(x, xp, fp):
#     # x: value to interpolate at
#     # xp: list of x positions (must be sorted)
#     # fp: list of y values at xp
#     if x <= xp[0]:
#         return fp[0]
#     if x >= xp[-1]:
#         return fp[-1]
#     for i in range(1, len(xp)):
#         if x < xp[i]:
#             x0, x1 = xp[i-1], xp[i]
#             y0, y1 = fp[i-1], fp[i]
#             return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
#     return fp[-1]

# # Convert distances to meters, velocities to m/s, powers to W
# distances_m = [d * 1000 for d in distances_final]
# velocities_mps = [v / 3.6 for v in velocities_final]
# powers_watt = [p * 1e6 for p in powers_final]
# s_grid = list(range(0, total_distance + delta_s, delta_s))
# v_opt = {}
# P_opt = {}
# for s in s_grid:
#     v_interp = linear_interpolate(s, distances_m, velocities_mps)
#     p_interp = linear_interpolate(s, distances_m, powers_watt)
#     v_opt[int(s)] = v_interp
#     P_opt[int(s)] = p_interp
# # Remove entries with keys greater than total_distance
# v_opt = {k: v for k, v in v_opt.items() if k <= total_distance}
# P_opt = {k: p for k, p in P_opt.items() if k <= total_distance}


# with open("vopt_popt.csv", "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Distance (m)", "Velocity (m/s)", "Power (W)"])
#     for s in sorted(v_opt.keys()):
#         writer.writerow([s, v_opt[s], P_opt[s]])
