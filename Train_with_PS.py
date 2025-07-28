''' Capability: This script sets up the environment for profile steering of multiple trains, a battery, and a PV system.
    It generates detailed train profiles using Train.py and includes the latest updates for iterative peak shaving,
    journey time extension adjustments, and flexibility management. The script now supports advanced adjustments such as
    reducing a train’s maximum power and extending journey times when needed to avoid excessive peak power, and it reflects
    these changes in both the simulation outputs and comparison plots.
    
- PV Generation Profiling:
  - Generates a realistic photovoltaic (PV) generation profile for the Netherlands based on time-of-day, weather conditions (sunny, partly cloudy, cloudy, rainy), and seasonal factors (using day of year).
  - Introduces realistic variability through sinusoidal and random variations to mimic cloud movements and atmospheric changes.

- Train Profile Generation:
  - Uses a detailed optimization model (via Pyomo in Train.py) to generate train power profiles based on speed limits, gradients, and other train parameters.
  - Converts distance-based results to time-based profiles using interpolation.

- Profile Steering Environment Setup:
  - Sets up an energy management environment that combines multiple devices (trains, PV system, battery).
  - Defines the simulation time horizon and creates desired power reference profiles (e.g., a zero or flat target profile).

- Device Modeling and Integration:
  - Models energy devices including battery, PV system, and train devices with corresponding power profiles.
  - Uses energy device classes (Battery, PV, TrainDevice) to simulate combined system behavior.

- Iterative Profile Steering Optimization:
  - Uses an iterative profile steering algorithm (via the profilesteering module) to converge the hub power profile toward the desired profile.
  - Incorporates constraints such as maximum iterations and convergence thresholds.

- Peak Shaving Capabilities:
  - Implements iterative peak shaving adjustments by reducing a train’s maximum power and/or extending journey times when peak power exceeds a set threshold.
  - Dynamically adjusts departure/arrival flexibility based on peak shaving iterations to flatten the power peaks.

- Scenario Comparison:
  - Provides a framework to run and compare multiple scenarios.
  - Allows different settings for time flexibility, train parameters, PV conditions, etc., and then plots a comparison of hub profiles as well as battery profiles.

- Visualization and Plotting:
  - Plots the generated PV profile.
  - Generates comparative plots for hub power profiles, battery profiles, and other devices’ profiles across different scenarios.
  - Provides detailed energy profile plots including battery State of Charge (SoC) and power contributions from each device.

- Robustness and Flexibility:
  - Includes error handling to capture infeasible journeys during the optimization.
  - Supports flexible input parameters for various aspects of the simulation (e.g., speed limits, delta_s, journey times, weather conditions).
'''
''' Inputs:
            In Train.py:
                        speed_limit_file_list: CSV files defining the route for different train journeys in terms of length and speed limits.
                        delta_s_list: Distance step sizes for each train (meters), affecting accuracy and computation time.
                        total_time_list: Total journey time for each train (seconds), which forms the basis for optimizing the driving strategy.
            In this file:
                        gradients: Specifies track gradients (default is a flat track). Adjust if necessary.
                        interpolation_step: Time-based interpolation step (recommended is 1 second) to convert Train.py’s distance-based results.
                        time_start_horizon and time_end_horizon: Define the simulation time horizon for profile steering.
                        desired_profile: A reference profile (typically zero flat) that matches the length of the extended time horizon.
                        num_trains: Number of trains simulated, each with its own profile.
                        start_times and end_times: Define the allowed time window for each train’s journey, based on scheduled times and flexibility margins.
                        generation_profile: PV generation profile; if None, a default flat profile is used.
                        Peak Shaving Adjustments: Parameters such as e_min, max_iters, peak_shaving_threshold, reduce_peak_power_for_shaving, and 
                             journey_time_extension_for_shaving enable iterative adjustments to reduce peak power by modifying train power 
                             and journey times while managing departure and arrival flexibilities.
            In battery.py:
                        capacity, max_power, min_power, initialSoC: Battery parameters (default set to specifications of Wierden’s battery).

''' 
''' Focusing on the battery in Wierden using the topological map:
        Wierden is connected to three substations—Almelo, Nijverdal, and Rijssen.
        When Wierden is identified as the hub, the effective battery power delivered is a function of the train’s
        distance from its associated substation. If ‘Wdn’ is not present in the filename, the departure substation
        power is calculated automatically based on the train distance across the route.
'''

from profilesteering import profilesteering
from profilesteering.dev.traindevice import TrainDevice
from profilesteering.dev.pv import PV
from profilesteering.dev.battery import Battery
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import contextlib
import os
sys.path.append(r"c:\Users\DarbandiH\OneDrive - University of Twente\Postdoc\Python\Train optimization problem")
from Train import Main, process_speed_limits, collecting_gradients, ElectricalParams, TrainParams, SimulationParams

def get_train_profiles(delta_s, total_time, speed_limit_file, interpolation_step, max_power):

    # Initialize train parameters
    electrical, train, simulation = ElectricalParams(), TrainParams(), SimulationParams()
    train.max_p = max_power
    simulation.delta_s = delta_s
    simulation.total_time = total_time
    simulation.time_remaining = simulation.total_time - simulation.t_init
    simulation.speed_limit_csv_path = r"c:\Users\DarbandiH\OneDrive - University of Twente\Postdoc\Python\Train optimization problem\SpeedLimits\\" + speed_limit_file
    
    # Process speed limits and gradients
    _, speed_limit_array, simulation.distance_remaining = process_speed_limits(simulation)
    gradients = collecting_gradients(simulation, mode="const", max_gradient=0.0)
    
    # Build data structure
    data = {0: {'grade': gradients[0], 'speed_limit': speed_limit_array[0]}}
    for i in range(1, int(simulation.distance_remaining / simulation.delta_s) + 1):
        data[i * simulation.delta_s] = {'grade': gradients[min(i, len(gradients) - 1)],
            'speed_limit': speed_limit_array[min(i, len(speed_limit_array) - 1)]}
    
    # Get the model results
    try:
        model, _ = Main(data, train, electrical, simulation)
    except RuntimeError as e:
        # Propagate the error up to be handled by the peak shaving loop
        raise RuntimeError(f"Train with {speed_limit_file} is infeasible: {e}")
    
    if "Wdn" in speed_limit_file:
        print(". Wierden hub energy is considered for this train.")
    else:
        print(". The departure substation energy is considered for this train.")
    # Original distance-to-time conversion
    original_times, original_train_power, original_train_traction, original_train_braking, original_hub_power = [], [], [], [], []
    for d in sorted(data.keys()):
        original_times.append(model.t[d]())
        original_train_power.append(model.Pg[d]() / 1e6)  # Convert to MW
        original_train_traction.append(model.Pm[d]() / 1e6)  # Convert to MW
        original_train_braking.append(model.Pn[d]() / 1e6)  # Convert to MW
        if "Wdn" in speed_limit_file:
            if speed_limit_file.startswith("SpeedLimit_Wdn"):
                original_hub_power.append(model.Pg[d]() / 1e6 * (simulation.distance_remaining - d) / simulation.distance_remaining)
            else:
                original_hub_power.append(model.Pg[d]() / 1e6 * d / simulation.distance_remaining)
        else: # automatically calculate the departure substation power based on distance instead of hub
            original_hub_power.append(model.Pg[d]() / 1e6 * (simulation.distance_remaining - d) / simulation.distance_remaining)

    # Create time points for interpolation
    time_start = int(original_times[0])
    time_end = int(original_times[-1])
    times = np.arange(time_start, time_end + 1, interpolation_step)
    
    # Create interpolation functions
    train_interpolator = interp1d(original_times, original_train_power, kind='linear', 
                                bounds_error=False, fill_value="extrapolate")
    hub_interpolator = interp1d(original_times, original_hub_power, kind='linear', 
                                bounds_error=False, fill_value="extrapolate")
    traction_interpolator = interp1d(original_times, original_train_traction, kind='linear',
                                   bounds_error=False, fill_value="extrapolate")
    braking_interpolator = interp1d(original_times, original_train_braking, kind='linear',
                                  bounds_error=False, fill_value="extrapolate")
    
    # Generate interpolated profiles
    train_overall = train_interpolator(times)
    hub_power = hub_interpolator(times)
    train_traction = traction_interpolator(times)
    train_braking = braking_interpolator(times)

    return hub_power, train_overall, train_traction, train_braking, times
def setup_profile_steering_environment(
    delta_s_list, 
    total_time_list, 
    speed_limit_file_list, 
    start_times, 
    end_times, 
    interpolation_step,
    time_start_horizon, 
    time_end_horizon, 
    train_max_p_list,
    generation_profile,
    start_time_hour, 
    max_capacity_mw, 
    weather_condition, 
    day_of_year, 
    random_seed
):
    """
    Sets up the environment for profile steering with multiple trains and energy devices.
    
    Args:
        delta_s_list (list): List of distance step sizes for each train (meters).
        total_time_list (list): List of total journey times for each train (seconds).
        speed_limit_file_list (list): List of speed limit files for each train.
        start_times (list): List of start times for each train in the time horizon (seconds).
        end_times (list): List of end times for each train in the time horizon (seconds).
        interpolation_step (int): Smoothness interval for interpolation (seconds).
        time_start_horizon (int): Start time for the extended time horizon (seconds).
        time_end_horizon (int or None): End time for the extended time horizon (seconds). If None, it is calculated.
        train_max_p_list (list): List of maximum power values for each train (Watts).
        generation_profile (list or None): PV generation profile (MW). If None, a default profile is used.
        
    Returns:
        tuple: (devices, desired_profile, extended_times)
            - devices: List of energy devices (Battery, PV, TrainDevices).
            - desired_profile: Desired power profile for the system (list of zeros).
            - extended_times: Time points for the extended time horizon (numpy array).
    """
    # Determine the number of trains from the input lists
    num_trains = len(delta_s_list)
    assert num_trains == len(total_time_list), "total_time_list must match the number of trains"
    assert num_trains == len(speed_limit_file_list), "speed_limit_file_list must match the number of trains"
    assert num_trains == len(start_times), "start_times must match the number of trains"
    assert num_trains == len(end_times), "end_times must match the number of trains"

    # Ensure start and end times are within the time horizon    
    start_times = [max(start, time_start_horizon) for start in start_times]
    if time_end_horizon is None:
        time_end_horizon = max(end_times)

    # Get train profiles for each train
    train_profiles = []
    
    for i in range(num_trains):
        hub_power, _, _, _, _ = get_train_profiles(
            delta_s=delta_s_list[i],
            total_time=total_time_list[i],
            speed_limit_file=speed_limit_file_list[i],
            interpolation_step=interpolation_step,
            max_power=train_max_p_list[i])
        train_profiles.append(hub_power.tolist())  
    
    # Define extended time horizon
    extended_times = np.arange(time_start_horizon, time_end_horizon + 1, 1)
    
    # Create desired profile (zeros) with the length of extended_times
    desired_profile = [0] * len(extended_times)

    # Create PV profile if not provided
    if generation_profile is None:
        generation_profile = [-0.2] * len(extended_times)  # Default constant generation
    else:
        generation_profile = generate_pv_profile(start_time_hour, len(extended_times), max_capacity_mw, 
                       weather_condition, day_of_year, random_seed=None)
    
    # Create devices (Battery, PV, and TrainDevices)
    devices = [Battery(), PV(generation_profile=generation_profile)]
    for i in range(num_trains):
        devices.append(TrainDevice(train_profiles[i], startTime=start_times[i], endTime=end_times[i]))
    
    return devices, desired_profile, extended_times, generation_profile
def plot_energy_profiles(extended_times, devices, battery, pv, soc):
    plt.figure(figsize=(12, 6))
    plt.plot(extended_times, pv.profile, label="PV Generation", color='yellow', linestyle='-')
    for i, device in enumerate(devices):
        if isinstance(device, TrainDevice):
            plt.plot(extended_times, device.profile, label=f"Train {i-1} Profile", linestyle=':', alpha=0.8)
    plt.plot(extended_times, battery.profile, label="Battery Profile", color='blue', linestyle='--')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(extended_times, soc[:-1], label="Battery SoC", color='red', linestyle='-')
    ax2.set_ylabel("State of Charge (SoC)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.title("Individual and Combined Profiles with Battery SoC")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Power (MW)")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
def plot_ps_vs_original(extended_times, final_profile, battery, pv):
    plt.figure(figsize=(12, 6))
    plt.plot(extended_times, final_profile, label="Hub with Battery and PV", color='green')
    plt.plot(extended_times, [f - b - p for f, b, p in zip(final_profile, battery.profile, pv.profile)], 
             label="Hub w/o Battery and PV", color='red')
    plt.plot(extended_times, pv.profile, label="PV Profile", color='grey', linestyle='-', alpha=0.8)
    plt.plot(extended_times, battery.profile, label="Battery Profile", color='grey', linestyle='--', alpha=0.8)
    plt.title("Hub Power Profile Optimization with Profile Steering")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Power (MW)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
def run_profile_steering(
    delta_s_list,
    speed_limit_file_list,
    train_max_p_list,
    interpolation_step,
    total_time_list,
    times,
    departure_flexibility,
    arrival_flexibility,
    time_start_horizon,
    time_end_horizon,
    e_min,
    max_iters,
    generation_profile,
    start_time_hour,
    max_capacity_mw,
    weather_condition,
    day_of_year, 
    random_seed,
    detailed_output_per_scenario,
    max_peak_shaving_iterations,
    peak_shaving_threshold,
    reduce_peak_power_for_shaving,
    journey_time_extension_for_shaving
):
    # Assert that journey time extension is less than the minimum flexibility    
    min_departure_flex = min(departure_flexibility) * 100  # Convert to percentage
    min_arrival_flex = min(arrival_flexibility) * 100  # Convert to percentage
    assert journey_time_extension_for_shaving <= min(min_departure_flex, min_arrival_flex), \
        f"Journey time extension ({journey_time_extension_for_shaving}%) exceeds available flexibility (min: {min(min_departure_flex, min_arrival_flex):.2f}%)"
    
    # Calculate start_times and end_times if not provided directly        
    start_times = [int(t * 60) - int(flex * total) 
                      for t, total, flex in zip(times, total_time_list, departure_flexibility)]
    end_times = [int(t * 60) + int((1 + flex) * total) 
                    for t, total, flex in zip(times, total_time_list, arrival_flexibility)]
    
    # Setup environment
    devices, desired_profile, extended_times, generation_profile = setup_profile_steering_environment(
        delta_s_list, total_time_list, speed_limit_file_list, start_times, end_times,
        interpolation_step, time_start_horizon, time_end_horizon, train_max_p_list, 
        generation_profile, start_time_hour, max_capacity_mw, weather_condition, day_of_year, random_seed)
    
    # Initialize and run profile steering
    ps = profilesteering.ProfileSteering(devices)
    ps.init(desired_profile)
    final_profile = ps.iterative(e_min, max_iters)
    
    # Perform peak shaving if needed
    final_profile, devices, extended_times, total_time_list, departure_flexibility, arrival_flexibility = perform_peak_shaving(
        final_profile, devices, delta_s_list, total_time_list, speed_limit_file_list,
        train_max_p_list, times, departure_flexibility, arrival_flexibility,
        start_times, end_times, interpolation_step, time_start_horizon, time_end_horizon,
        generation_profile, start_time_hour, max_capacity_mw, weather_condition, day_of_year, random_seed, 
        e_min, max_iters, 
        max_peak_shaving_iterations, peak_shaving_threshold, reduce_peak_power_for_shaving, journey_time_extension_for_shaving
    )
    
    # Extract device profiles
    battery = devices[0]
    pv = devices[1]

    if detailed_output_per_scenario:
        soc = battery.calculate_soc(times=extended_times, return_history=True)
        plot_energy_profiles(extended_times, devices, battery, pv, soc)
        plot_ps_vs_original(extended_times, final_profile, battery, pv)
        plt.show()
    
    return final_profile, battery.profile, extended_times
def perform_peak_shaving(
    final_profile,
    devices,
    delta_s_list,
    total_time_list, 
    speed_limit_file_list,
    train_max_p_list,
    times,
    departure_flexibility,
    arrival_flexibility,
    start_times,
    end_times,
    interpolation_step,
    time_start_horizon,
    time_end_horizon,
    generation_profile,
    start_time_hour,
    max_capacity_mw,
    weather_condition,
    day_of_year,
    random_seed,
    e_min,
    max_iters,
    max_peak_shaving_iterations,
    peak_shaving_threshold,
    reduce_peak_power_for_shaving,
    journey_time_extension_for_shaving
):
    """
    Perform peak shaving on the power profile by iteratively adjusting train power or journey times.
    
    Args:
        final_profile: Current power profile to be peak-shaved
        devices: List of devices in the system (battery, PV, trains)
        delta_s_list: List of distance steps for trains
        total_time_list: List of journey times for each train
        speed_limit_file_list: List of speed limit files for trains
        train_max_p_list: List of maximum power values for trains
        times: Scheduled times for trains (in minutes)
        departure_flexibility: Flexibility in departure times
        arrival_flexibility: Flexibility in arrival times
        start_times: Calculated start times for trains
        end_times: Calculated end times for trains
        interpolation_step: Time step for interpolation
        time_start_horizon: Start time of simulation horizon
        time_end_horizon: End time of simulation horizon
        generation_profile: PV generation profile
        e_min: Convergence threshold for profile steering
        max_iters: Maximum iterations for profile steering
        max_peak_shaving_iterations: Maximum iterations for peak shaving
        peak_shaving_threshold: Power threshold for peak shaving
        reduce_peak_power_for_shaving: Percentage to reduce power
        journey_time_extension_for_shaving: Percentage to extend journey time
        
    Returns:
        tuple: 
            - final_profile: Updated power profile
            - devices: Updated devices
            - extended_times: Time points for extended horizon
            - total_time_list: Updated journey times
            - departure_flexibility: Updated departure flexibility
            - arrival_flexibility: Updated arrival flexibility
    """
    if max(final_profile) <= peak_shaving_threshold or max_peak_shaving_iterations is None:
        # No peak shaving needed
        extended_times = np.arange(time_start_horizon, max(end_times) + 1, 1)
        return final_profile, devices, extended_times, total_time_list, departure_flexibility, arrival_flexibility
    else:
        iteration = 0
        train_adjustment_counts = {}  # Track how many times each train has been adjusted
        previous_peak_power = float('inf')  # Track previous iteration's peak power
        extended_times = np.arange(time_start_horizon, max(end_times) + 1, 1)

        while (max(final_profile) > peak_shaving_threshold and 
            iteration < max_peak_shaving_iterations):
            
            iteration += 1
            peak_time = np.argmax(final_profile)
            peak_power = final_profile[peak_time]
            
            # Check if we've made progress since the last iteration
            if peak_power >= previous_peak_power and iteration > 1:
                print(f"\nNo improvement in peak power after iteration {iteration-1}, stopping peak shaving")
                break
                
            previous_peak_power = peak_power
            print(f"\nPeak Shaving Iteration {iteration}: Peak power {peak_power:.3f} MW at time {peak_time}s (threshold: {peak_shaving_threshold} MW)")
            
            # Find trains active during peak time
            active_trains = []
            infeasible_trains = set()
            
            for i, device in enumerate(devices[2:], 2):  # Skip battery and PV devices
                train_idx = i-2
                if device.startTime <= peak_time <= device.endTime:
                    active_trains.append((train_idx, device, train_max_p_list[train_idx]))
            
            if not active_trains:
                print(f"No more adjustable trains active at peak time {peak_time}s, cannot reduce further")
                break
            
            # Try adjusting trains until we find one that works
            adjustment_successful = False
            while active_trains and not adjustment_successful:
                # Select train with highest max power
                max_power_train = max(active_trains, key=lambda x: x[2])
                selected_train_idx, selected_train, original_max_power = max_power_train
                
                # Store original time values before any adjustments
                original_total_time = total_time_list[selected_train_idx]
                
                # Adjust the profile of the selected train
                train_max_p_list[selected_train_idx] *= (1 - reduce_peak_power_for_shaving/100)
                
                # Update adjustment count display
                current_count = train_adjustment_counts.get(selected_train_idx, 0) + 1
                print(f"Reducing max power of Train {selected_train_idx+1} ({speed_limit_file_list[selected_train_idx][11:18]}) by {reduce_peak_power_for_shaving}% (adjustment #{current_count})")
                
                # First try - only reduce power
                try:
                    with open(os.devnull, 'w') as fnull:
                        with contextlib.redirect_stdout(fnull):
                            devices, desired_profile, extended_times, generation_profile = setup_profile_steering_environment(
        delta_s_list, total_time_list, speed_limit_file_list, start_times, end_times,
        interpolation_step, time_start_horizon, time_end_horizon, train_max_p_list, 
        generation_profile, start_time_hour, max_capacity_mw, weather_condition, day_of_year, random_seed)
                            
                            ps = profilesteering.ProfileSteering(devices)
                            ps.init(desired_profile)
                            final_profile = ps.iterative(e_min, max_iters)
                    
                    new_peak_power = max(final_profile)
                    print(f"After adjustment: Peak power {new_peak_power:.3f} MW")
                    
                    if new_peak_power >= peak_power:
                        print("No improvement in peak power after adjustment, trying another train")
                        # Restore original power and try another train
                        train_max_p_list[selected_train_idx] = original_max_power
                        active_trains.remove(max_power_train)
                        infeasible_trains.add(selected_train_idx)
                    else:
                        # Successful adjustment - update the count
                        train_adjustment_counts[selected_train_idx] = current_count
                        adjustment_successful = True
                        
                except RuntimeError as e:
                    print(f"Train {selected_train_idx+1} journey infeasible with reduced power: {e}")
                    
                    # Second try - extend journey time and try again with reduced power
                    print(f"Trying to extend journey time by {journey_time_extension_for_shaving}% for Train {selected_train_idx+1}")
                    
                    # Extend journey time
                    total_time_list[selected_train_idx] *= 1 + journey_time_extension_for_shaving / 100
                    
                    # Recalculate start and end times
                    start_times = [int(t * 60) - int(flex * total) 
                                for t, total, flex in zip(times, total_time_list, departure_flexibility)]
                    end_times = [int(t * 60) + int((1 + flex) * total) 
                                for t, total, flex in zip(times, total_time_list, arrival_flexibility)]
                    
                    try:
                        with open(os.devnull, 'w') as fnull:
                            with contextlib.redirect_stdout(fnull):
                                devices, desired_profile, extended_times, generation_profile = setup_profile_steering_environment(
        delta_s_list, total_time_list, speed_limit_file_list, start_times, end_times,
        interpolation_step, time_start_horizon, time_end_horizon, train_max_p_list, 
        generation_profile, start_time_hour, max_capacity_mw, weather_condition, day_of_year, random_seed)
                            
                                ps = profilesteering.ProfileSteering(devices)
                                ps.init(desired_profile)
                                final_profile = ps.iterative(e_min, max_iters)
                    
                        new_peak_power = max(final_profile)
                        print(f"After power reduction and time extension: Peak power {new_peak_power:.3f} MW")
                        
                        if new_peak_power >= peak_power:
                            print("No improvement in peak power after power+time adjustment, trying another train")
                            # Restore original values and try another train
                            train_max_p_list[selected_train_idx] = original_max_power
                            total_time_list[selected_train_idx] = original_total_time
                            active_trains.remove(max_power_train)
                            infeasible_trains.add(selected_train_idx)
                        else:
                            # Successful adjustment with extended time
                            train_adjustment_counts[selected_train_idx] = current_count
                            adjustment_successful = True
                            
                            # Reduce departure and arrival flexibility when journey time extension is accepted
                            flex_reduction = journey_time_extension_for_shaving / 2 / 100  # Half of the journey time extension percentage
                            
                            # Store original flexibility values for logging
                            original_dep_flex = departure_flexibility[selected_train_idx]
                            original_arr_flex = arrival_flexibility[selected_train_idx]
                            
                            # Reduce both flexibilities
                            departure_flexibility[selected_train_idx] = max(0, departure_flexibility[selected_train_idx] - flex_reduction)
                            arrival_flexibility[selected_train_idx] = max(0, arrival_flexibility[selected_train_idx] - flex_reduction)
                            
                            print(f"Successfully adjusted Train {selected_train_idx+1} with both reduced power and extended journey time ({int(original_total_time)}s → {int(total_time_list[selected_train_idx])}s)")
                            print(f"Reduced flexibilities: departure {original_dep_flex:.3f} → {departure_flexibility[selected_train_idx]:.3f}, " +
                                f"arrival {original_arr_flex:.3f} → {arrival_flexibility[selected_train_idx]:.3f}")
                            
                    except RuntimeError as e:
                        print(f"Train {selected_train_idx+1} journey still infeasible after time extension: {e}")
                        # Restore original values and try another train
                        train_max_p_list[selected_train_idx] = original_max_power
                        total_time_list[selected_train_idx] = original_total_time
                        active_trains.remove(max_power_train)
                        infeasible_trains.add(selected_train_idx)
        
        # If we tried all trains and none worked
        if not adjustment_successful:
            print("Peak shaving is infeasible - all trains at peak time would become infeasible with reduced power")
            print("Continuing with the last valid profile")
        
        return final_profile, devices, extended_times, total_time_list, departure_flexibility, arrival_flexibility
def compare_scenarios(scenario_params_list, scenario_labels):
    """
    Compare hub and battery profiles from multiple scenarios.
    
    Args:
        scenario_params_list (list): List of dictionaries, each containing parameters for a scenario.
        scenario_labels (list): List of labels for each scenario.
    
    Raises:
        AssertionError: If the number of scenarios does not match the number of labels or if no scenarios are provided.
    """

    # Validate input
    assert len(scenario_params_list) == len(scenario_labels), "Number of scenarios must match number of labels"
    assert len(scenario_params_list) > 0, "At least one scenario must be provided"
    
    # Set up colors for different scenarios
    colors = ['green', 'blue', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    # If we have more scenarios than colors, colors will cycle
    
    # Run all scenarios and collect results
    final_profiles, battery_profiles, time_profiles = [], [], []
    
    for i, (scenario, label) in enumerate(zip(scenario_params_list, scenario_labels)):
        print(f"Running {label}...")
        final, battery, times = run_profile_steering(**scenario)
        final_profiles.append(final)
        battery_profiles.append(battery)
        time_profiles.append(times)
    
    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # Plot hub profiles
    for i, (final, times, label) in enumerate(zip(final_profiles, time_profiles, scenario_labels)):
        color = colors[i % len(colors)]
        ax1.plot(times, final, label=label, color=color)
    ax1.set_title("Hub Power Profile Comparison")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Power (MW)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # Plot battery profiles
    for i, (battery, times, label) in enumerate(zip(battery_profiles, time_profiles, scenario_labels)):
        color = colors[i % len(colors)]
        ax2.plot(times, battery, label=label, color=color)
    ax2.set_title("Battery Profile Comparison")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Power (MW)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.show()
def generate_pv_profile(start_time_hour, duration_seconds, max_capacity_mw, 
                       weather_condition, day_of_year, random_seed=None):
    """
    Generate a realistic PV generation profile based on time of day and weather conditions for the Netherlands.
    
    Args:
        start_time_hour (int or float): Hour of day when the simulation starts (0-23)
        duration_seconds (int): Duration of the simulation in seconds
        max_capacity_mw (float): Maximum capacity of the PV installation in MW
        weather_condition (str): Weather condition - "sunny", "partly_cloudy", "cloudy", "rainy"
        day_of_year (int): Day of year (1-365) for seasonal adjustments
        random_seed (int): Seed for reproducible randomization, None for random
        
    Returns:
        list: PV generation profile in MW for each second of the simulation
    """
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Determine seasonal factor (1.0 in summer, 0.4 in winter)
    summer_peak = 172  # June 21st
    winter_peak = 355  # December 21st
    days_from_winter = min((day_of_year - winter_peak) % 365, (winter_peak - day_of_year) % 365)
    seasonal_factor = 0.4 + 0.6 * (days_from_winter / 182.5)
    
    # Weather condition factors
    weather_factors = {
        "sunny": 1.0,
        "partly_cloudy": 0.7,
        "cloudy": 0.4,
        "rainy": 0.2
    }
    weather_factor = weather_factors.get(weather_condition.lower(), 0.7)
    
    # Generate time points for the simulation duration
    time_points = np.arange(0, duration_seconds)
    
    # Generate output profile
    pv_profile = []
    
    # Calculate sun parameters for the Netherlands
    sunrise_hour = 5.5 + (1 - seasonal_factor) * 2  # Earlier in summer (4-5 AM), later in winter (7-8 AM)
    sunset_hour = 21.0 - seasonal_factor * 3        # Later in summer (21-22 PM), earlier in winter (17-18 PM)
    daylight_hours = sunset_hour - sunrise_hour
    peak_hour = sunrise_hour + daylight_hours / 2   # Solar noon
    
    # Process each second of the simulation
    for second in time_points:
        # Convert simulation time to hour of day
        current_hour = (start_time_hour + second / 3600) % 24
        
        # Calculate solar output (bell curve based on time of day)
        if sunrise_hour <= current_hour <= sunset_hour:
            # Distance from solar noon normalized to a -1 to 1 range
            normalized_time = 2 * (current_hour - peak_hour) / daylight_hours
            
            # Bell curve formula: e^(-x²)
            sun_position_factor = np.exp(-(normalized_time ** 2) * 4)
            
            # Add some realistic variations for clouds/atmosphere
            variation = 1.0 + 0.1 * np.sin(second / 300) + 0.05 * np.sin(second / 120)
            
            # Add small random variations (1-minute cloud movements)
            if second % 60 == 0:
                cloud_variation = 0.95 + 0.1 * np.random.random()
            else:
                cloud_variation = 1.0
                
            # Calculate output with all factors combined
            output = max_capacity_mw * sun_position_factor * weather_factor * seasonal_factor * variation * cloud_variation
            
            # Ensure output doesn't exceed capacity or go negative
            output = max(0, min(output, max_capacity_mw))
            
            # Generate negative profile for consumption convention
            pv_profile.append(-output)
        else:
            # No generation during night
            pv_profile.append(0)
    
    return pv_profile
##################################################################################
#Scenario compare part 
##################################################################################
scenario1 = {
    # Train.py settings
    'delta_s_list': [100] * 10, # Distance step for each train in meters (delta_s in Train.py)
    'speed_limit_file_list': ["SpeedLimit_Alm_Wdn.csv", "SpeedLimit_Wdn_Alm.csv", "SpeedLimit_Wdn_Nij.csv", 
        "SpeedLimit_Alm_Rij.csv", "SpeedLimit_Alm_Rij.csv", "SpeedLimit_Nij_Wdn.csv", 
        "SpeedLimit_Alm_Wdn.csv", "SpeedLimit_Rij_Wdn.csv", "SpeedLimit_Wdn_Alm.csv", 
        "SpeedLimit_Wdn_Rij.csv"], # Speed limit files for each train
    'train_max_p_list': [2159400]*10, # Maximum power capability of trains in Watts.
    
    # Converting Train.py outputs from distance to time
    'interpolation_step': 1,
    
    # Timetable settings
    'total_time_list': [270, 270, 330, 460, 460, 330, 270, 315, 270, 315], # Total journey time for each train in seconds
    'times': [0, 1, 6, 6, 11, 16, 20, 22, 22, 25], # Scheduled times in minutes (the hashtags in --:##) of an hour
    'departure_flexibility': [0.03]*10, # margin or flexibility for departure times
    'arrival_flexibility': [0.03]*10,# margin or flexibility for arrival times
    
    # Profile steering settings
    'time_start_horizon': 0,
    'time_end_horizon': None, # None means calculate based on start and end times
    'e_min': 0.001, # Minimum error for convergence in profile steering
    'max_iters': 1000, # Maximum iterations for profile steering

    # PV generation profile
    'generation_profile' : 1, # None means default PV profile (posstivie flat profile)
    'start_time_hour': 12, # Start time of the simulation in hours (0-23)
    'max_capacity_mw': 5.0, # Maximum capacity of the PV installation in MW
    'weather_condition': 'rainy', # Weather condition - 'sunny', 'partly_cloudy', 'cloudy', 'rainy'
    'day_of_year': 172, # Day of year (1-365) for seasonal adjustments
    'random_seed': 1, # Seed for reproducible randomization, None for random

    # Whether to return detailed output data and create plots per scenario
    'detailed_output_per_scenario': True,
    
    # For peak shaving (Note: These parameters are only used if peak shaving is enabled in the profile steering)
    'max_peak_shaving_iterations': None, # Maximum iterations for peak shaving, None means no peak shaving avoidance
    'peak_shaving_threshold': 0.25, # The threshold for peak shaving, cut/flatten the peaks above MW
    'reduce_peak_power_for_shaving': 5, # Percentage to reduce peak power of the train that triggered peak shaving
    'journey_time_extension_for_shaving': 1 # Percentage to extend journey time of the train that triggered peak shaving
}

scenario2 = {
    # Train.py settings
    'delta_s_list': [100] * 10, # Distance step for each train in meters (delta_s in Train.py)
    'speed_limit_file_list': ["SpeedLimit_Alm_Wdn.csv", "SpeedLimit_Wdn_Alm.csv", "SpeedLimit_Wdn_Nij.csv", 
        "SpeedLimit_Alm_Rij.csv", "SpeedLimit_Alm_Rij.csv", "SpeedLimit_Nij_Wdn.csv", 
        "SpeedLimit_Alm_Wdn.csv", "SpeedLimit_Rij_Wdn.csv", "SpeedLimit_Wdn_Alm.csv", 
        "SpeedLimit_Wdn_Rij.csv"], # Speed limit files for each train
    'train_max_p_list': [2159400]*10, # Maximum power capability of trains in Watts.
    
    # Converting Train.py outputs from distance to time
    'interpolation_step': 1,
    
    # Timetable settings
    'total_time_list': [270, 270, 330, 460, 460, 330, 270, 315, 270, 315], # Total journey time for each train in seconds
    'times': [0, 1, 6, 6, 11, 16, 20, 22, 22, 25], # Scheduled times in minutes (the hashtags in --:##) of an hour
    'departure_flexibility': [0.03]*10, # margin or flexibility for departure times
    'arrival_flexibility': [0.03]*10,# margin or flexibility for arrival times
    
    # Profile steering settings
    'time_start_horizon': 0,
    'time_end_horizon': None, # None means calculate based on start and end times
    'e_min': 0.001, # Minimum error for convergence in profile steering
    'max_iters': 1000, # Maximum iterations for profile steering

    # PV generation profile
    'generation_profile' : 1, # None means default PV profile (posstivie flat profile)
    'start_time_hour': 12, # Start time of the simulation in hours (0-23)
    'max_capacity_mw': 5.0, # Maximum capacity of the PV installation in MW
    'weather_condition': 'rainy', # Weather condition - 'sunny', 'partly_cloudy', 'cloudy', 'rainy'
    'day_of_year': 172, # Day of year (1-365) for seasonal adjustments
    'random_seed': 1, # Seed for reproducible randomization, None for random

    # Whether to return detailed output data and create plots per scenario
    'detailed_output_per_scenario': True,
    
    # For peak shaving (Note: These parameters are only used if peak shaving is enabled in the profile steering)
    'max_peak_shaving_iterations': 10, # Maximum iterations for peak shaving, None means no peak shaving avoidance
    'peak_shaving_threshold': 0.25, # The threshold for peak shaving, cut/flatten the peaks above MW
    'reduce_peak_power_for_shaving': 5, # Percentage to reduce peak power of the train that triggered peak shaving
    'journey_time_extension_for_shaving': 1 # Percentage to extend journey time of the train that triggered peak shaving
}


if __name__ == "__main__":
    try:
        compare_scenarios([scenario1,scenario2],
            ["No peak shaving", "With peak shaving"])
        
    except AssertionError as e:
        print(f"\n{e}")  # This will only print the assertion message
    except RuntimeError as e:
        print(f"\nProfile steering stopped due to: {e}")
        print()