import pyomo.environ as pyomo
import matplotlib.pyplot as plt
import sys
import os
import random
import time
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import csv  # <-- add this import

def generate_random_gradients(distance_remaining, delta_s, mode, max_gradient):
    # Generate gradients for the track profile. mode: "const" for constant gradient, "randm" for random profile.
    num_steps = int(distance_remaining // delta_s + 1)
    if mode == "const":
        gradients = [max_gradient] * num_steps
    elif mode == "randm":
        gradients = [random.uniform(-max_gradient, max_gradient)]
        for i in range(1, num_steps):
            prev_gradient = gradients[-1]
            new_gradient = prev_gradient + random.uniform(-0.0001, 0.0001)
            new_gradient = max(-max_gradient, min(max_gradient, new_gradient))
            gradients.append(new_gradient)
    else:
        raise ValueError("mode must be 'const' or 'randm'")
    return gradients

def generate_random_track_radius(distance_remaining, delta_s, mode, min_radius, max_radius):
    # Generate track radius profile. mode: "const" for constant radius, "randm" for random profile.
    num_steps = int(distance_remaining // delta_s + 1)
    if mode == "const":
        track_radius = [max_radius] * num_steps
    elif mode == "randm":
        track_radius = [random.uniform(min_radius, max_radius)]
        for i in range(1, num_steps):
            prev_radius = track_radius[-1]
            # Small random walk, but keep within bounds
            new_radius = prev_radius + random.uniform(-20, 20)
            new_radius = max(min_radius, min(max_radius, new_radius))
            track_radius.append(new_radius)
    else:
        raise ValueError("mode must be 'const' or 'randm'")
    return track_radius

def Initializer(delta_s, max_acc, max_braking, data, m, C_d, A, C, eta, WindSpeed, v_init):
    D = data.keys()  # Distance steps
    model0 = pyomo.ConcreteModel()
    model0.v = pyomo.Var(D, domain=pyomo.NonNegativeReals, bounds=(0, max_v))  # Velocity (m/s)
    model0.P = pyomo.Var(D, domain=pyomo.Reals)  # Power consumption (kW)
    model0.of = pyomo.Objective(expr=0.0, sense=pyomo.minimize)
    model0.cons = pyomo.ConstraintList()
    
    final_distance = list(D)[-1]
    # model0.cons.add(model0.s[final_time] == distance_remaining)
    model0.v[0].fix(v_init)  # Initial velocity is 0
    model0.v[final_distance].fix(0)  # Final velocity is 0
    model0.P[0].fix(0)


    for d in list(D)[1:]:
        prev_d = d - delta_s
        model0.cons.add((model0.v[d] - model0.v[prev_d]) / (2 * delta_s / (model0.v[d] + model0.v[prev_d])) <= max_acc)
        model0.cons.add(-(model0.v[d] - model0.v[prev_d]) / (2 * delta_s / (model0.v[d] + model0.v[prev_d])) <= max_braking)

        # Simpler version of Davies equation for power consumption
        model0.cons.add(model0.P[d] == 1 / eta * (
            0.5 * 1.225 * C_d * A * (model0.v[d] + WindSpeed)**2+
            C * m * 9.807 +
            m * 9.807 * data[d]['grade'] +  # Gradient at this distance
            m * (model0.v[d] - model0.v[prev_d]) / (2 * delta_s / (model0.v[d] + model0.v[prev_d]))
        ) * model0.v[d])

    solver = pyomo.SolverFactory('ipopt')
    results = solver.solve(model0, tee=False)
    if results.solver.termination_condition == pyomo.TerminationCondition.optimal:
        v_opt = {d: model0.v[d].value for d in D}
        P_opt = {d: model0.P[d].value for d in D}
    else:
        print(f"Infeasible during Initialization - {results.solver.termination_condition}")
        sys.exit()
    return v_opt, P_opt

def train(distance_remaining, delta_s, max_acc, max_braking, max_p, data, m, C_d, A, C, eta, braking_eff, time_remaining, WindSpeed, v_init, max_p_sub1, max_p_sub2, mu_curve, Consider_electrical_losses, rho, V0):
    start_train = time.time()
    v_opt, P_opt = Initializer(delta_s, max_acc, max_braking, data, m, C_d, A, C, eta, WindSpeed, v_init)
    D = data.keys()
    model = pyomo.ConcreteModel()

    # Decision Variables
    model.v = pyomo.Var(D, domain=pyomo.NonNegativeReals, bounds=(0, max_v), initialize=lambda model0, d: v_opt[d]) # velocity (m/s)

    # State Variables
    model.Pm = pyomo.Var(D, domain=pyomo.NonNegativeReals, bounds=(0, max_p)) 
    model.Pn = pyomo.Var(D, domain=pyomo.NonNegativeReals) 
    model.P = pyomo.Var(D, domain=pyomo.Reals, initialize=lambda model0, d: P_opt[d])
    model.t = pyomo.Var(D, domain=pyomo.NonNegativeReals) # Distance 
    
    # Objective Function
    model.of = pyomo.Objective(expr=sum(model.Pm[d] - model.Pn[d] * braking_eff for d in D), sense=pyomo.minimize)
    
    # Constraints
    model.cons = pyomo.ConstraintList()
    final_distance = list(D)[-1]
    
    # Initial conditions
    model.v[0].fix(v_init)
    model.Pn[0].fix(0)
    model.v[final_distance].fix(0)
    model.t[final_distance].fix(time_remaining)
    
    for d in list(D)[1:]:
        prev_d = d - delta_s
        model.cons.add(model.t[d] == model.t[prev_d] + 2 * delta_s / (model.v[d] + model.v[prev_d]))
        model.cons.add(model.v[d] <= data[d]['speed_limit'])  # Speed limit constraint
        model.cons.add((model.v[d] - model.v[prev_d]) / (2 * delta_s / (model.v[d] + model.v[prev_d])) <= max_acc)
        model.cons.add(-(model.v[d] - model.v[prev_d]) / (2 * delta_s / (model.v[d] + model.v[prev_d])) <= max_braking)        
        model.cons.add(model.P[d] == model.Pm[d] - model.Pn[d])
        model.cons.add(abs(model.P[d])<= distance_remaining*max_p_sub1/(distance_remaining - d + 1e-9))  # Avoid division by zero
        model.cons.add(abs(model.P[d])<= distance_remaining *max_p_sub2/(d + 1e-9))  # Avoid division by zero
        
        # Davies equation for power consumption + Electrical losses
        model.cons.add(
            model.P[d] == model.v[d] / eta * ( #mechanical losses
            0.5 * 1.225 * C_d * A * (model.v[d] + WindSpeed)**2 +
            C * m * 9.807 +
            m * 9.807 * data[d]['grade'] +
            m * (model.v[d] - model.v[prev_d]) / (2 * delta_s / (model.v[d] + model.v[prev_d])) +
            mu_curve * m * (model.v[d]**2) / data[d]['radius']
            ) + rho * distance_remaining * Consider_electrical_losses * ( #electrical losses
            (
                model.v[d] / eta * (
                0.5 * 1.225 * C_d * A * (model.v[d] + WindSpeed)**2 +
                C * m * 9.807 +
                m * 9.807 * data[d]['grade'] +
                m * (model.v[d] - model.v[prev_d]) / (2 * delta_s / (model.v[d] + model.v[prev_d])) +
                mu_curve * m * (model.v[d]**2) / data[d]['radius']
                )
            ) / V0
            )**2
        )
        
    solver = pyomo.SolverFactory('ipopt')
    results = solver.solve(model, tee=False)
    end_train = time.time()
    print(f"Training time: {end_train - start_train:.2f} seconds")
    return model, results.solver.termination_condition  # Add return statement  # Add return statement

def plot_Pm_and_Pn_profile_time(model, data):
    times = []
    Pm_values = []
    Pn_values = []
    accelerations = []
    velocities = []

    for d in data.keys():
        times.append(model.t[d]())  # Get time at each distance step
        Pm_values.append(model.Pm[d]() / 1000000)  # Convert W to MW
        Pn_values.append(model.Pn[d]() / 1000000)  # Convert W to MW
        v = model.v[d]()
        velocities.append(v * 3.6)  # Convert m/s to km/h

        # Calculate acceleration
        if d == 0:
            a = 0
        else:
            prev_d = d - delta_s
            a = (model.v[d]() - model.v[prev_d]()) / (2 * delta_s / (model.v[d]() + model.v[prev_d]()))
        accelerations.append(a)

    # Create the plot with three y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot power on primary y-axis
    ax1.plot(times, Pm_values, 'b-', linewidth=2, label='Positive Power (Pm)')
    ax1.plot(times, Pn_values, 'r-', linewidth=2, label='Negative Power (Pn)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Power (MW)', color='b')

    # Create secondary y-axis for acceleration
    ax2 = ax1.twinx()
    ax2.plot(times, accelerations, 'g-', linewidth=2, label='Acceleration')
    ax2.set_ylabel('Acceleration (m/s²)', color='g')

    # Create third y-axis for velocity
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.plot(times, velocities, 'orange', linewidth=2, label='Velocity')
    ax3.set_ylabel('Velocity (km/h)', color='orange')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')

    plt.title(f'Train Power, Acceleration, and Velocity Profile (Time-Based)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)   

def plot_Pm_and_Pn_profile(model, data, speed_limits=None):
    distances = []
    Pm_values = []
    Pn_values = []
    accelerations = []
    velocities = []

    for d in data.keys():
        distances.append(d / 1000)  # Convert distance to km
        Pm_values.append(model.Pm[d]() / 1000000)  # Convert W to MW
        # print(model.Pm[d]()/1000000)
        Pn_values.append(model.Pn[d]() / 1000000)  # Convert W to MW
        v = model.v[d]()
        velocities.append(v * 3.6)  # Convert m/s to km/h

        # Calculate acceleration
        if d == 0:
            a = 0
        else:
            prev_d = d - delta_s
            a = (model.v[d]() - model.v[prev_d]()) / (2 * delta_s / (model.v[d]() + model.v[prev_d]()))

        accelerations.append(a)

    # Create the plot with three y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot power on primary y-axis
    ax1.plot(distances, [pm - pn for pm, pn in zip(Pm_values, Pn_values)], 'b-', linewidth=2, label='Power (P)')
    ax1.axhline(0, color='black', linewidth=1, linestyle='-')  # Add horizontal line at y=0
    ax1.axvline(0, color='black', linewidth=2)  # y-axis at x=0
    ax1.set_xlim(left=0)
    ax1.set_xlabel('Distance (km)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Power (MW)', color='b', fontsize=16, fontweight='bold')

    # Add more grid lines for y-axis only
    ax1.yaxis.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.xaxis.grid(False)

    # Add more ticks for y axis
    y_major_locator = MultipleLocator(0.2)  # Major ticks every 0.2 MW
    y_minor_locator = AutoMinorLocator(4)   # 4 minor ticks between majors
    ax1.yaxis.set_major_locator(y_major_locator)
    ax1.yaxis.set_minor_locator(y_minor_locator)

    # Create secondary y-axis for acceleration
    ax2 = ax1.twinx()
    ax2.plot(distances, accelerations, 'g-', linewidth=2, label='Acceleration')
    ax2.set_ylabel('Acceleration (m/s²)', color='g', fontsize=16, fontweight='bold')

    # Create third y-axis for velocity
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.plot(distances, velocities, 'orange', linewidth=2, label='Velocity')
    ax3.set_ylabel('Velocity (km/h)', color='orange', fontsize=16, fontweight='bold')

    # Overlay speed limit profile from CSV if provided
    if speed_limits is not None:
        # Prepare step plot data for speed limits
        interval_starts = list(speed_limits.keys())
        interval_ends = [speed_limits[d]['end'] for d in interval_starts]
        interval_speeds = [speed_limits[d]['speed'] for d in interval_starts]
        plot_distances = []
        plot_speeds = []
        for i in range(len(interval_starts)):
            if i == 0:
                plot_distances.append(interval_starts[i] / 1000)
                plot_speeds.append(interval_speeds[i])
            plot_distances.append(interval_ends[i] / 1000)
            plot_speeds.append(interval_speeds[i + 1] if i + 1 < len(interval_speeds) else interval_speeds[i])
        # Plot on the velocity axis (ax3)
        ax3.step(plot_distances, plot_speeds, where='post', color='red', linewidth=2, linestyle='--', label='MRSP')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')

    minutes = int(time_remaining // 60)
    seconds = int(time_remaining % 60)
    plt.title(f'Train Power, Acceleration, and Velocity Profile (S={distance_remaining/1000} km, Run time={minutes} min {seconds} sec)', fontsize=18,fontweight='bold')
    
def plot_substation_powers(model, data):
    distances = []
    P_sub1_values = []
    P_sub2_values = []

    for d in data.keys():
        distances.append(d / 1000)  # Convert distance to km
        P = model.P[d]()
        P_sub1 = (distance_remaining - d) * P / distance_remaining
        P_sub2 = d * P / distance_remaining
        P_sub1_values.append(P_sub1 / 1000000)  # Convert W to MW
        P_sub2_values.append(P_sub2 / 1000000)  # Convert W to MW

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(distances, P_sub1_values, 'b-', linewidth=2, label='Substation 1 Power')
    ax1.plot(distances, P_sub2_values, 'r-', linewidth=2, label='Substation 2 Power')
    ax1.axhline(0, color='black', linewidth=1, linestyle='-')  # Add horizontal line at y=0
    ax1.axvline(0, color='black', linewidth=2)  # y-axis at x=0
    ax1.set_xlim(left=0)
    ax1.set_xlabel('Distance (km)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Power (MW)', color='b', fontsize=16, fontweight='bold')

    # Add more grid lines for y-axis only
    ax1.yaxis.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.xaxis.grid(False)

    # Add more ticks for y axis
    y_major_locator = MultipleLocator(0.2)  # Major ticks every 0.2 MW
    y_minor_locator = AutoMinorLocator(4)   # 4 minor ticks between majors
    ax1.yaxis.set_major_locator(y_major_locator)
    ax1.yaxis.set_minor_locator(y_minor_locator)

    minutes, seconds = int(time_remaining // 60), int(time_remaining % 60)
    plt.title(f'Substation Power Profile (S={distance_remaining/1000} km, Run time={minutes} min {seconds} sec)', fontsize=18, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=14)
    
def plot_voltage_profile(model, data):
    distances = []
    voltages = []

    for d in data.keys():
        distances.append(d / 1000)  # Convert distance to km

        # Calculate voltage using the given formula
        if d == 0:
            voltages.append(V0)  # Initial voltage
        else:
            P = model.P[d]()
            V = V0 - P / ((V0 / (rho * d + 1e-9)) + (V0 / (rho * (distance_remaining - d + 1e-9))))
            voltages.append(V)

    plt.figure(figsize=(12, 6))
    plt.plot(distances, voltages, 'b-', linewidth=2)
    plt.xlabel('Distance (km)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Train Voltage Profile (S={distance_remaining/1000} km)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

def plot_distance_vs_time(model, data):
    # Extract time and distance data
    times = []
    distances = []
    
    for d in data.keys():
        # Get time in seconds
        times.append(model.t[d]())
        # Get distance in kilometers
        distances.append(d / 1000)  # Convert distance to km
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, distances, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance (km)')
    plt.title(f'Train Distance vs. Time (S={distance_remaining/1000} km)')
    
    # Add horizontal and vertical gridlines
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
def plot_gradients_vs_distance(data, distance_remaining, delta_s):
    # Extract distance and gradient data
    distances = []
    gradients = []
    
    for d in data.keys():
        distances.append(d / 1000)  # Convert distance to km
        gradients.append(data[d]['grade'])  # Gradient values
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(distances, gradients, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Distance (km)')
    plt.ylabel('Gradient')
    plt.title(f'Train Gradient Profile (S={distance_remaining/1000} km, Step={delta_s} m)')
    plt.ylim(-0.005, 0.005)  # Set y-axis limits to keep it constant
    
    # Add horizontal and vertical gridlines
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
        
def calculate_Pm_per_d(model, data):
    # Create a dictionary to store the Pm for each distance step
    Pm_per_d = {}
    
    for d in data.keys():
        Pm_value = model.Pm[d]() / 1000000  # Convert Pm from W to MW
        Pm_per_d[d] = Pm_value

    # Print the results
    print("Pm per distance step (in MW):")
    for distance, Pm in sorted(Pm_per_d.items()):
        print(f"Distance {distance}m: {Pm:.6f} MW")

    # Calculate the average Pm over all distance steps
    total_Pm = sum(Pm_per_d.values()) / len(Pm_per_d)
    print(f"\nAverage Pm over all distance steps: {total_Pm:.6f} MW")
    print(f"Total distance steps: {len(Pm_per_d)}")

    return Pm_per_d, total_Pm

def calculate_energy_consumption(model, data, delta_s):
    total_energy = 0  # Initialize total energy in kWh
    for d in list(data.keys())[1:]:
        prev_d = d - delta_s
        avg_power = (model.P[d]() + model.P[prev_d]()) / 2 # Calculate average power between two points (in W)
        if model.v[d]() + model.v[prev_d]() > 0:
            delta_t = 2 * delta_s / (model.v[d]() + model.v[prev_d]())
        else:
            delta_t = 0
        if avg_power < 0:
            avg_power = avg_power * braking_eff  # Adjust for regenerative braking
        total_energy += (avg_power * delta_t) / 3.6e6         # Energy for this segment in kWh
    return total_energy

def calculate_total_curve_energy(model, data, mu_curve, m, delta_s):
    total_curve_energy = 0  # in kWh
    for d in list(data.keys())[1:]:
        prev_d = d - delta_s
        radius = data[d]['radius'] if data[d]['radius'] != 0 else 1e-9
        v_avg = (model.v[d]() + model.v[prev_d]()) / 2
        total_curve_energy += mu_curve * m * (v_avg ** 3) / radius / 3.6e6  # in kWh
    return total_curve_energy

def process_speed_limits(filepath, delta_s, d_init):
    """
    Reads a CSV file generated from Rijstrategie MSRP and processes speed limits for every delta_s step.
    The first and second zero-speed rows are used to determine the speed limits.
    Returns:
        - speed_limits_dict: Dictionary for plotting speed limits
        - speed_limit_array: Array of speed limits for each delta_s step
    """
    # Read the CSV file
    distances = []
    speeds = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        header = next(reader)  # Skip header
        for row in reader:
            try:
                distance = float(row[1])
                speed = float(row[2])
                distances.append(distance)
                speeds.append(speed)
            except (IndexError, ValueError):
                continue

    # Find the first and second zero speed rows
    zero_indices = [i for i, s in enumerate(speeds) if s == 0]
    if len(zero_indices) < 2:
        raise ValueError("CSV does not contain at least two zero-speed rows.")

    start_idx = zero_indices[0]
    end_idx = zero_indices[1]
    while d_init>=distances[start_idx]:
        start_idx += 1
        if start_idx >= len(distances):
            raise ValueError("Initial distance exceeds the range of the speed limits in the CSV file.")
    start_distance = distances[start_idx]
    end_distance = distances[end_idx]
    distance_remaining = end_distance - start_distance
    print(f"Start distance: {start_distance} m, End distance: {end_distance} m")

    # Build speed limits dictionary for plotting
    speed_limits_dict = {}
    for i in range(start_idx, end_idx):
        d_start = distances[i] - start_distance  # Shift distances to start from 0
        d_end = distances[i+1] - start_distance if i+1 <= end_idx else end_distance - start_distance
        speed = speeds[1] if i == 0 else speeds[i]
        speed_limits_dict[d_start] = {'end': d_end, 'speed': speed}

    # Generate speed limit array for every delta_s step
    num_steps = int(distance_remaining // delta_s + 1)
    speed_limit_array = []
    sorted_limits = sorted(speed_limits_dict.items())

    for i in range(num_steps):
        current_distance = i * delta_s
        # Find applicable speed limit
        for (start_d, limit_data) in sorted_limits:
            if start_d <= current_distance < limit_data['end']:
                speed_limit_array.append(limit_data['speed'] / 3.6)  # Convert km/h to m/s
                break
        else:
            # If no interval found, use the last known speed limit
            speed_limit_array.append(sorted_limits[-1][1]['speed'] / 3.6)

    return speed_limits_dict, speed_limit_array, distance_remaining


# Electrical Parameters
rho, V0 = 0.00003, 1500 # Ohms/m, Voltage (V)
Consider_electrical_losses = 0 # Electrical losses in Train function (0: do not consider, 1: consider)

# Train Parameters (SNG)
Rotatory_inertia_factor = 0.0674
m = 152743 * (1 + Rotatory_inertia_factor) # kg (train weight, but should be variable later)
A, C, C_d = 2.88*4.25, 0.002, 0.8 # m^2 (Frontal area), (Rolling resistance coefficient), (Drag coefficient)
eta = 0.857  # Efficiency of the train's propulsion system
braking_eff = 0.0  # Regenerative braking efficiency
max_v, max_acc, max_braking = 44.444, 0.81, 0.5 # m/s = 160 km/h (max velocity), m/s2 (max acceleration), (max braking)
max_p = 1393000 # W (max power)
mu_curve = 0.001 # Curve resistance coefficient (m/s^2) (assumed value, can be adjusted based on specific conditions)

# Distance discretization
total_time = 400 # (sec) From Substation 1 to Substation 2
delta_s = 50  # Distance step in meters

# Time-dependent parameters
max_p_sub1 = 100.0 * 1000000 # W (max power for substation 1)
max_p_sub2 = 100.0 * 1000000 # W (max power for substation 2)
# WindSpeed = random.choice([random.uniform(-5, -2), random.uniform(2, 5)])  # m/s (Wind speed, excluding -2 to 2)
WindSpeed = 0

# Initialize initial conditions
v_init = 0 / 3.6 # (Enter in km/h) Initial velocity (m/s)
t_init = 0 * 60 # Initial time (s)
d_init = 0 * 1000 # Initial distance (m)
time_remaining = total_time - t_init # Remaining time (s), Remaining distance (m)

speed_limit_csv_path = r"c:\Users\DarbandiH\OneDrive - University of Twente\Postdoc\Python\Train optimization problem\Scenario1_SLT.csv"
speed_limits_dict, speed_limit_array, distance_remaining = process_speed_limits(speed_limit_csv_path, delta_s, d_init)
print(f"Distance remaining: {distance_remaining} m")

gradients = generate_random_gradients(distance_remaining, delta_s, mode = "const", max_gradient=0.0)
track_radius = generate_random_track_radius(distance_remaining, delta_s, "const", min_radius=1000, max_radius=1e9)

data = {0: {'grade': gradients[0], 'radius': track_radius[0], 'speed_limit': speed_limit_array[0]}}

for i in range(1, int(distance_remaining / delta_s) + 1):
    data[i * delta_s] = {
        'grade': gradients[min(i, len(gradients) - 1)],
        'radius': track_radius[min(i, len(track_radius) - 1)],
        'speed_limit': speed_limit_array[min(i, len(speed_limit_array) - 1)]
    }
  
model, termination_condition = train(distance_remaining, delta_s, max_acc, max_braking, max_p, data, m, C_d, A, C, eta, braking_eff, time_remaining, WindSpeed, v_init, max_p_sub1, max_p_sub2, mu_curve, Consider_electrical_losses, rho, V0)
end_time = time.time()


def save_power_velocity_acceleration_to_csv(filepath, model, data, delta_s):
    """
    Saves power, velocity, and acceleration data in terms of distance to a CSV file.
    """
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Distance (km)", "Velocity (km/h)", "Power (MW)", "Acceleration (m/s²)"])  # Header
        for d in data.keys():
            distance_km = d / 1000  # Convert distance to km
            velocity_kmh = model.v[d]() * 3.6  # Convert velocity to km/h
            power_mw = model.P[d]() / 1e6  # Convert power to MW
            if d == 0:
                acceleration = 0  # No acceleration at the start
            else:
                prev_d = d - delta_s
                acceleration = (model.v[d]() - model.v[prev_d]()) / (2 * delta_s / (model.v[d]() + model.v[prev_d]()))
            writer.writerow([f"{distance_km:.3f}", f"{velocity_kmh:.3f}", f"{power_mw:.3f}", f"{acceleration:.3f}"])

if termination_condition in [pyomo.TerminationCondition.optimal, pyomo.TerminationCondition.locallyOptimal]:
    # Save results to file
    # base_filename = f"t{time_remaining}_S{distance_remaining}"
    # ext = ".txt"
    # suffix_num = 0
    # filename = f"{base_filename}_e{suffix_num}{ext}"
    # while os.path.exists(filename):
    #     suffix_num += 1
    #     filename = f"{base_filename}_e{suffix_num}{ext}"

    # with open(filename, 'w') as f:
    #     f.write(f"P_sub1_max: {max_p_sub1} W, P_sub2_max: {max_p_sub2} W\n")
    #     f.write(f"Braking Efficiency: {braking_eff*100}%\n")
    #     f.write("Distance(m), Velocity(km/h), Power (MW), P_sub1 (MW), P_sub2 (MW)\n")  # Header
    #     for d in data.keys():
    #         v_out = 3.6 * model.v[d]()
    #         if model.P[d]() < 0:
    #             P_out = braking_eff * model.P[d]() / 1000000  # Added braking_eff for braking power
    #         else:
    #             P_out = model.P[d]() / 1000000
    #         # P_sub1_out = model.P_sub1[d]() / 1000000
    #         # P_sub2_out = model.P_sub2[d]() / 1000000
    #         # f.write(f"{d}, {v_out:.3f}, {P_out:.6f}, {P_sub1_out:.6f}, {P_sub2_out:.6f}\n")

    # Plot results
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 18, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14, 'figure.titlesize': 20, 'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Computer Modern Roman', 'DejaVu Serif', 'serif']})
    print(f"\nTotal Energy Consumption: {calculate_energy_consumption(model, data, delta_s):.3f} kWh")
    print(f"\nTotal curve resistance sum over journey: {calculate_total_curve_energy(model, data, mu_curve, m, delta_s):.3f} kWh")
    # plot_substation_powers(model, data)
    plot_Pm_and_Pn_profile(model, data, speed_limits=speed_limits_dict)
    # save_power_velocity_acceleration_to_csv("Train_Results.csv", model, data, delta_s)
    # print("Results saved to 'Train_Results.csv'.")
    plt.show()
else:
    print("No results to save or plot due to solver termination condition.")