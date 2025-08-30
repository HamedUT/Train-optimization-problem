import pyomo.environ as pyomo
import matplotlib.pyplot as plt
import sys
import random
import time
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import csv
from dataclasses import dataclass

# === Parameter Classes ===
@dataclass
class ElectricalParams:
    rho: float = 0.00003  # Ohms/m
    V0: int = 1800        # Operational Voltage (V)
    cat_voltage_bound: list = (1200, 1925)  # [min, max] catenary voltage
    Consider_electrical_losses: int = 0     # 0: do not consider, 1: consider (but maybe it is double counting considering the constraint with voltage losses)
    max_p_sub1: float = 10.0e6       # W
    max_p_sub2: float = 10.0e6       # W
@dataclass
class TrainParams:
    m: float = 391000               # kg
    A: float = 3.02 * 4.67          # m^2
    C: float = 0.002                # Rolling resistance coefficient
    C_d: float = 0.8                # Drag coefficient
    eta: float = 1               # Propulsion efficiency
    braking_eff: float = 0.9       # Regenerative braking efficiency
    max_v: float = 44.444           # m/s
    max_acc: float = 0.768          # m/s^2
    max_braking: float = 0.5        # m/s^2
    max_p: float = 359900 * 6       # W
    mu_curve: float = 0.001         # Curve resistance coefficient
@dataclass
class SimulationParams:
    total_time: float = int(360)    # sec
    delta_s: int =       100        # m
    WindSpeed: float = 0            # m/s
    v_init: float = 0 / 3.6         # m/s
    t_init: float = 0 * 60          # s
    d_init: float = 0 * 1000        # m
    speed_limit_csv_path = r"c:\Users\DarbandiH\OneDrive - University of Twente\Postdoc\Python\Train optimization problem\SpeedLimits\SpeedLimit_Alakiii.csv"

electrical = ElectricalParams()
train = TrainParams()
simulation = SimulationParams()
simulation.time_remaining = simulation.total_time - simulation.t_init

def main():
    speed_limits_dict, speed_limit_array, simulation.distance_remaining = process_speed_limits(simulation)
    gradients = collecting_gradients(simulation, mode = "const", max_gradient=0.0)
    data = {0: {'grade': gradients[0], 'speed_limit': speed_limit_array[0]}}
    for i in range(1, int(simulation.distance_remaining / simulation.delta_s) + 1):
        data[i * simulation.delta_s] = {'grade': gradients[min(i, len(gradients) - 1)],
            'speed_limit': speed_limit_array[min(i, len(speed_limit_array) - 1)]}
    
    start_analyze = time.time()
    model, termination_condition = Main(data, train, electrical, simulation)
    end_analyze = time.time()
    print(f"Training time: {end_analyze - start_analyze:.2f} seconds")

    if termination_condition in [pyomo.TerminationCondition.optimal, pyomo.TerminationCondition.locallyOptimal]:
        plt.rcParams.update({'font.size': 15, 'axes.titlesize': 18, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14, 'figure.titlesize': 20, 'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Computer Modern Roman', 'DejaVu Serif', 'serif']})
        plot_substation_powers(model, data, simulation)
        plot_Pm_and_Pn_profile(model, data, simulation, speed_limits=speed_limits_dict)
        plot_Pm_and_Pn_profile_time(model, data, simulation, speed_limits=speed_limits_dict)
        plot_voltage_profile(model, data, simulation)
        save_power_velocity_acceleration_to_csv("Train_results.csv", model, data, simulation, calculate_energy_consumption(model, data, simulation, print_results=True))
        plt.show()
    else:
        print("No results to save or plot due to solver termination condition.")

    # Uncomment the following lines to run multiple scenarios with different max_p_sub1 values
    # max_p_sub1_list = [1e6, 1.5e6, 2.5e6]  # Different maximum power values for substation 1
    # plot_multiple_scenarios(max_p_sub1_list)

'Generate track gradients profile. Modes: "const" for constant gradient, "randm" for random profile, "files" for extracting from a file.'
def collecting_gradients(simulation: SimulationParams, mode, max_gradient): 
    num_steps = int(simulation.distance_remaining // simulation.delta_s + 1)
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

def Initializer(data, train: TrainParams, simulation: SimulationParams): #Provides the main model with initial profiles for power and speed. If not, the main model diverges.
    D = data.keys()  # Distance steps
    model0 = pyomo.ConcreteModel()
    model0.v = pyomo.Var(D, domain=pyomo.NonNegativeReals, bounds=(0, train.max_v))  # Velocity (m/s)
    model0.P = pyomo.Var(D, domain=pyomo.Reals)  # Power consumption (kW)
    model0.of = pyomo.Objective(expr=0.0, sense=pyomo.minimize)
    model0.cons = pyomo.ConstraintList()
    
    final_distance = list(D)[-1]
    model0.v[0].fix(simulation.v_init)  # Initial velocity is 0
    model0.v[final_distance].fix(0)  # Final velocity is 0
    model0.P[0].fix(0)

    for d in list(D)[1:]:
        prev_d = d - simulation.delta_s
        model0.cons.add((model0.v[d] - model0.v[prev_d]) / (2 * simulation.delta_s / (model0.v[d] + model0.v[prev_d]+1e-6)) <= train.max_acc)
        model0.cons.add(-(model0.v[d] - model0.v[prev_d]) / (2 * simulation.delta_s / (model0.v[d] + model0.v[prev_d]+1e-6)) <= train.max_braking)
        # Simpler version of Davies equation for power consumption
        model0.cons.add(model0.P[d] == model0.v[d] * (10 * (model0.v[d])**2 + 98070 +
            1e6 * (model0.v[d] - model0.v[prev_d]) / (2 * simulation.delta_s / (model0.v[d] + model0.v[prev_d]+1e-6))))

    solver = pyomo.SolverFactory('ipopt')
    results = solver.solve(model0, tee=False)
    if results.solver.termination_condition == pyomo.TerminationCondition.optimal:
        v_opt = {d: model0.v[d].value for d in D}
        P_opt = {d: model0.P[d].value for d in D}
    else:
        print(f"Infeasible during Initialization - {results.solver.termination_condition}")
        raise RuntimeError(f"Infeasible during Initialization - {results.solver.termination_condition}")
    return v_opt, P_opt

def Main(data, train: TrainParams, electrical: ElectricalParams, simulation: SimulationParams):
    v_opt, P_opt = Initializer(data, train, simulation)
    D = data.keys()
    model = pyomo.ConcreteModel()

    # Decision Variables
    model.v = pyomo.Var(D, domain=pyomo.NonNegativeReals, bounds=(0, train.max_v), initialize=lambda model0, d: v_opt[d]) # velocity (m/s)

    # State Variables
    model.Pm = pyomo.Var(D, domain=pyomo.NonNegativeReals, bounds=(0, train.max_p))
    model.Pn = pyomo.Var(D, domain=pyomo.NonNegativeReals) 
    model.Pg = pyomo.Var(D, domain=pyomo.Reals, initialize=lambda model0, d: P_opt[d])
    model.t = pyomo.Var(D, domain=pyomo.NonNegativeReals) 
    model.V = pyomo.Var(D, domain=pyomo.NonNegativeReals, bounds=(electrical.cat_voltage_bound[0], electrical.cat_voltage_bound[1])) # Voltage (according to ProRail safety standards)

    model.of = pyomo.Objective(expr=sum(model.Pg[d] for d in list(D)[1:]),sense=pyomo.minimize)
    
    # Constraints
    model.cons = pyomo.ConstraintList()
    final_distance = list(D)[-1]
    model.v[0].fix(simulation.v_init)
    model.Pm[0].fix(0)
    model.Pn[0].fix(0)
    model.v[final_distance].fix(0)
    model.t[final_distance].fix(simulation.time_remaining)
    model.V[0].fix(electrical.V0)
    model.Pg[0].fix(0)
    
    for d in list(D)[1:]:
        model.cons.add(model.t[d] == model.t[d - simulation.delta_s] + 2 * simulation.delta_s / (model.v[d] + model.v[d - simulation.delta_s]+1e-6))
        model.cons.add(model.v[d] <= data[d]['speed_limit'])  # Track speed limit constraint
        model.cons.add((model.v[d] - model.v[d - simulation.delta_s]) / (2 * simulation.delta_s / (model.v[d] + model.v[d - simulation.delta_s]+1e-6)) <= train.max_acc)
        model.cons.add(-(model.v[d] - model.v[d - simulation.delta_s]) / (2 * simulation.delta_s / (model.v[d] + model.v[d - simulation.delta_s]+1e-6)) <= train.max_braking)
        model.cons.add(model.Pg[d] == model.Pm[d]/train.eta - model.Pn[d]*train.braking_eff)
        model.cons.add(model.Pm[d] - model.Pn[d] == 
                model.v[d] * (0.5 * 1.225 * train.C_d * train.A * (model.v[d] + simulation.WindSpeed)**2 + # aerodynamic drag
                train.C * train.m * 9.807 + # rolling resistance
                train.m * 9.807 * data[d]['grade'] + # track grade
                train.m * (model.v[d] - model.v[d - simulation.delta_s]) / (2 * simulation.delta_s / (model.v[d] + model.v[d - simulation.delta_s]+1e-6))) # acceleration
            + electrical.rho * simulation.distance_remaining * electrical.Consider_electrical_losses * # electrical losses, but maybe it is double counting considering the constraint with voltage losses
                (model.v[d] / model.V[d] * (
                0.5 * 1.225 * train.C_d * train.A * (model.v[d] + simulation.WindSpeed)**2 +
                train.C * train.m * 9.807 +
                train.m * 9.807 * data[d]['grade'] +
                train.m * (model.v[d] - model.v[d - simulation.delta_s]) / (2 * simulation.delta_s / (model.v[d] + model.v[d - simulation.delta_s]))
                ))**2)
        model.cons.add(model.Pg[d] == model.V[d] * ((electrical.V0 - model.V[d]) / (electrical.rho * d + 1e-9) + (electrical.V0 - model.V[d]) / (electrical.rho * (simulation.distance_remaining - d) + 1e-9)))  # Electrical power consumption
        model.cons.add(abs(model.Pg[d]) <= (simulation.distance_remaining*electrical.max_p_sub1/(simulation.distance_remaining - d + 1e-9)))  # Avoid division by zero
        model.cons.add(abs(model.Pg[d]) <= (simulation.distance_remaining *electrical.max_p_sub2/(d + 1e-9)))  # Avoid division by zero

    solver = pyomo.SolverFactory('ipopt')
    solver.options['tol'] = 1e-8
    results = solver.solve(model, tee=False)
    if results.solver.termination_condition not in [pyomo.TerminationCondition.optimal, pyomo.TerminationCondition.locallyOptimal]:
        raise RuntimeError(f"Infeasible during Main solve - {results.solver.termination_condition}")
    return model, results.solver.termination_condition 

def plot_Pm_and_Pn_profile_time(model, data, simulation: SimulationParams, speed_limits=None):
    
    times, Pm_values, Pn_values, velocities, accelerations = [0], [0], [0], [0], [0]    
    for idx, d in enumerate(data.keys()):
        if idx == 0:
            continue
        t = model.t[d]()
        v = model.v[d]()
        pm = model.Pm[d]() / 1e6  # Convert to MW
        pn = model.Pn[d]() / 1e6  
        vel_kmh = v * 3.6  # Convert velocity to km/h
        
        times.append(t)
        Pm_values.append(pm)
        Pn_values.append(pn)
        velocities.append(vel_kmh)

        # Calculate acceleration
        prev_d = d - simulation.delta_s
        if prev_d in model.v:
            a = (v - model.v[prev_d]()) / (2 * simulation.delta_s / (v + model.v[prev_d]() + 1e-6))
        else:
            a = 0  # Default acceleration if previous distance is not available
        accelerations.append(a)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(times, [pm / train.eta - train.braking_eff * pn for pm, pn in zip(Pm_values, Pn_values)],
             'b-', linewidth=2, label='Train Electrical Power')
    ax1.axhline(0, color='black', linewidth=1, linestyle='-')
    ax1.axvline(0, color='black', linewidth=2)
    ax1.set_xlim(left=0)
    ax1.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Power (MW)', color='b', fontsize=16, fontweight='bold')
    ax1.yaxis.grid(True, which='major', linestyle='--', alpha=0.5)
    ax1.xaxis.grid(True, which='major', linestyle='--', alpha=0.5)
    y_major_locator = MultipleLocator(0.5)  # every 0.2 MW
    ax1.yaxis.set_major_locator(y_major_locator)

    ax2 = ax1.twinx()
    ax2.plot(times, velocities, 'orange', linewidth=2, label='Velocity')
    ax2.set_ylabel('Velocity (km/h)', color='orange', fontsize=16, fontweight='bold')
    if speed_limits is not None: # Overlay speed limit profile from CSV if provided
        interval_starts = list(speed_limits.keys())
        interval_ends = [speed_limits[d]['end'] for d in interval_starts]
        interval_speeds = [speed_limits[d]['speed'] for d in interval_starts]
        plot_times, plot_speeds = [], []
        for i in range(len(interval_starts)):
            d_start, d_end = interval_starts[i], interval_ends[i]
            d_keys = list(data.keys())
            t_start = model.t[d_keys[min(range(len(d_keys)), key=lambda j: abs(d_keys[j] - d_start))]]()
            t_end = model.t[d_keys[min(range(len(d_keys)), key=lambda j: abs(d_keys[j] - d_end))]]()
            if i == 0:
                plot_times.append(t_start)
                plot_speeds.append(interval_speeds[i])
            plot_times.append(t_end)
            plot_speeds.append(interval_speeds[i + 1] if i + 1 < len(interval_speeds) else interval_speeds[i])
        # Plot speed limit
        ax2.step(plot_times, plot_speeds, where='post', color='#b58900', linewidth=2, linestyle='--', label='Speed Limit')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    minutes, seconds = int(simulation.time_remaining // 60), int(simulation.time_remaining % 60)
    total_energy = calculate_energy_consumption(model, data, simulation, print_results=False)
    plt.title(
        f'Train Power and Velocity Profile (S={simulation.distance_remaining / 1000} km, '
        f'Run time={minutes} min {seconds} sec, Energy={total_energy:.3f} kWh)',
        fontsize=18, fontweight='bold')

def plot_Pm_and_Pn_profile(model, data, simulation: SimulationParams, speed_limits=None):
    distances, power_values, Pg_values, Pb_values, velocities, accelerations = [0], [0], [0], [0], [0], [0]

    d_keys = list(data.keys())
    for idx in range(1, len(d_keys)):
        d, prev_d = d_keys[idx], d_keys[idx - 1]
        distances.append(d / 1000)
        power_values.append(1e-6*(model.Pm[d]()/train.eta - model.Pn[d]()* train.braking_eff))  # Convert to MW
        Pg_values.append(1e-6*model.Pg[d]()) #power from other sources (grid or substation)
        velocities.append(model.v[d]() * 3.6)
        accelerations.append((model.v[d]() - model.v[prev_d]()) / (2 * simulation.delta_s / (model.v[d]() + model.v[prev_d]() + 1e-6)))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(distances, power_values, 'b-', linewidth=2, label='Train Electrical Power')
    ax1.axhline(0, color='black', linewidth=1, linestyle='-')
    ax1.axvline(0, color='black', linewidth=2)
    ax1.set_xlim(left=0)
    ax1.set_xlabel('Distance (km)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Power (MW)', color='b', fontsize=16, fontweight='bold')
    y_major_locator = MultipleLocator(0.5)
    ax1.yaxis.set_major_locator(y_major_locator)
    ax1.yaxis.grid(True, which='major', linestyle='--', alpha=0.5)
    ax1.xaxis.grid(True, which='major', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(distances, velocities, 'orange', linewidth=2, label='Velocity')
    ax2.set_ylabel('Velocity (km/h)', color='orange', fontsize=16, fontweight='bold')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax1.yaxis.set_label_position('left')
    ax1.yaxis.tick_left()

    if speed_limits is not None:
        interval_starts = list(speed_limits.keys())
        interval_ends = [speed_limits[d]['end'] for d in interval_starts]
        interval_speeds = [speed_limits[d]['speed'] for d in interval_starts]
        plot_distances, plot_speeds = [interval_starts[0] / 1000],[interval_speeds[0]]
        for i in range(len(interval_starts)):
            d_start = interval_starts[i]
            d_end = interval_ends[i]
            plot_distances.append(d_end / 1000)
            plot_speeds.append(interval_speeds[i + 1] if i + 1 < len(interval_speeds) else interval_speeds[i])
        ax2.step(plot_distances, plot_speeds, where='post', color='#b58900', linewidth=2, linestyle='--', label='Speed Limit')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    minutes = int(simulation.time_remaining // 60)
    seconds = int(simulation.time_remaining % 60)
    total_energy= calculate_energy_consumption(model, data, simulation, print_results=False)
    plt.title(
        f'Train Power and Velocity Profile (S={simulation.distance_remaining/1000} km, Run time={minutes} min {seconds} sec, Energy={total_energy:.3f} kWh)',
        fontsize=18, fontweight='bold')
    
def plot_substation_powers(model, data, simulation: SimulationParams):
    distances, P_sub1_values, P_sub2_values, SoC_values = [], [], [], []
    d_keys = list(data.keys())
    for idx in range(1, len(d_keys)):
        d = d_keys[idx]
        distances.append(d / 1000)
        P_sub1_values.append((simulation.distance_remaining - d) * model.Pg[d]() * 1e-6 / simulation.distance_remaining) 
        P_sub2_values.append(d * model.Pg[d]() * 1e-6 / simulation.distance_remaining)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(distances, P_sub1_values, 'b-', linewidth=2, label='Substation 1 Power')
    ax1.plot(distances, P_sub2_values, 'r-', linewidth=2, label='Substation 2 Power')
    ax1.axhline(0, color='black', linewidth=1, linestyle='-')
    ax1.axvline(0, color='black', linewidth=2)
    ax1.set_xlim(left=0)
    ax1.set_xlabel('Distance (km)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Power (MW)', color='b', fontsize=16, fontweight='bold')
    y_major_locator = MultipleLocator(0.25)
    ax1.yaxis.set_major_locator(y_major_locator)
    ax1.yaxis.grid(True, which='major', linestyle='--', alpha=0.5)
    ax1.xaxis.grid(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right', fontsize=14)

    minutes, seconds = int(simulation.time_remaining // 60), int(simulation.time_remaining % 60)
    plt.title(f'Substation Power Profile (S={simulation.distance_remaining/1000} km, Run time={minutes} min {seconds} sec)', fontsize=18, fontweight='bold')
    
def plot_voltage_profile(model, data, simulation: SimulationParams):
    distances, voltages = [], []

    for d in data.keys():
        distances.append(d / 1000)  # Convert distance to km
        voltages.append(model.V[d]())  # Use voltage from the model

    plt.figure(figsize=(12, 6))
    plt.plot(distances, voltages, 'b-', linewidth=2)
    plt.xlabel('Distance (km)', fontsize=16, fontweight='bold')
    plt.ylabel('Voltage (V)', fontsize=16, fontweight='bold')
    plt.title(f'Train Voltage Profile (S={simulation.distance_remaining/1000} km)', fontsize=18, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

def plot_distance_vs_time(model, data, simulation: SimulationParams):
    # Extract time and distance data
    times, distances = [], []
    
    for d in data.keys():
        times.append(model.t[d]())
        distances.append(d / 1000)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, distances, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance (km)')
    plt.title(f'Train Distance vs. Time (S={simulation.distance_remaining/1000} km)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
def plot_gradients_vs_distance(data, simulation: SimulationParams):
    distances, gradients = [],[]
    
    for d in data.keys():
        distances.append(d / 1000)  # Convert distance to km
        gradients.append(data[d]['grade'])  # Gradient values
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(distances, gradients, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Distance (km)')
    plt.ylabel('Gradient')
    plt.title(f'Train Gradient Profile (S={simulation.distance_remaining/1000} km, Step={delta_s} m)')
    plt.ylim(-0.005, 0.005)  # Set y-axis limits to keep it constant
    
    # Add horizontal and vertical gridlines
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

def calculate_energy_consumption(model, data, simulation: SimulationParams, print_results=True):
    Train_energy_consumption = 0
    for d in list(data.keys())[1:]:
        prev_d = d - simulation.delta_s
        if model.v[d]() + model.v[prev_d]() > 0:
            delta_t = 2 * simulation.delta_s / (model.v[d]() + model.v[prev_d]())
        else:
            delta_t = 0
        Pm_avg = (model.Pm[d]() + model.Pm[prev_d]()) / 2
        Pn_avg = (model.Pn[d]() + model.Pn[prev_d]()) / 2
        Train_energy_consumption += (Pm_avg/train.eta - Pn_avg*train.braking_eff) * delta_t / 3.6e6
    if print_results:
        print(f"Mechanical energy consumption: {Train_energy_consumption:.3f} kWh") 
    return Train_energy_consumption

def process_speed_limits(simulation: SimulationParams):
    """
    Reads a CSV file with distance and speed limit rows.
    The speed limit in each row applies from that distance up to the next row's distance.
    The last row's speed limit is ignored.
    Returns:
        - speed_limits_dict: Dictionary for plotting speed limits
        - speed_limit_array: Array of speed limits for each delta_s step
        - simulation.distance_remaining: Total distance covered by the speed limits
    """
    # Read the CSV file
    distances = []
    speeds = []
    with open(simulation.speed_limit_csv_path, newline='', encoding='utf-8') as csvfile:
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

    if len(distances) < 2:
        raise ValueError("CSV must contain at least two rows for speed limits.")

    # Use all rows, each speed applies from its distance to the next
    start_distance = distances[0]
    end_distance = distances[-1]
    simulation.distance_remaining = end_distance - start_distance
    filename = os.path.basename(simulation.speed_limit_csv_path)
    print(f"Distance: {int(end_distance-start_distance)} m, Route: {filename[:-4][-7:]}, Step: {simulation.delta_s} m, Time: {simulation.time_remaining} s",end=" ")

    # Build speed limits dictionary for plotting
    speed_limits_dict = {}
    for i in range(len(distances) - 1):
        d_start = distances[i] - start_distance  # Shift distances to start from 0
        d_end = distances[i + 1] - start_distance
        speed = speeds[i]
        speed_limits_dict[d_start] = {'end': d_end, 'speed': speed}

    # Generate speed limit array for every delta_s step
    num_steps = int(simulation.distance_remaining // simulation.delta_s + 1)
    speed_limit_array = []
    sorted_limits = sorted(speed_limits_dict.items())

    for i in range(num_steps):
        current_distance = i * simulation.delta_s
        # Find applicable speed limit
        for (start_d, limit_data) in sorted_limits:
            if start_d <= current_distance < limit_data['end']:
                speed_limit_array.append(limit_data['speed'] / 3.6)  # Convert km/h to m/s
                break
        else:
            # If no interval found, use the last known speed limit
            speed_limit_array.append(sorted_limits[-1][1]['speed'] / 3.6)
    return speed_limits_dict, speed_limit_array, simulation.distance_remaining

def save_power_velocity_acceleration_to_csv(filepath, model, data, simulation: SimulationParams, total_energy=None):
    """
    Saves power, velocity, acceleration, and time data in terms of distance to a CSV file.
    Optionally appends total energy consumption at the end.
    """
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Distance (km)", "Time (s)", "Velocity (km/h)", "Power (MW)", "Acceleration (m/s²)"])  # Header
        for d in data.keys():
            distance_km = d / 1000  # Convert distance to km
            time_s = model.t[d]()   # Time at this distance step
            velocity_kmh = model.v[d]() * 3.6  # Convert velocity to km/h
            power_mw = (model.Pg[d]()) / 1e6  # Convert power to MW
            if d == 0:
                acceleration = 0  # No acceleration at the start
            else:
                prev_d = d - simulation.delta_s
                acceleration = (model.v[d]() - model.v[prev_d]()) / (2 * simulation.delta_s / (model.v[d]() + model.v[prev_d]()))
            writer.writerow([f"{distance_km:.3f}", f"{time_s:.2f}", f"{velocity_kmh:.3f}", f"{power_mw:.3f}", f"{acceleration:.3f}"])
        # Append total energy at the end if provided
        if total_energy is not None:
            writer.writerow([])
            writer.writerow(["Total Energy Consumption (kWh):" f"{total_energy:.3f}"])

def plot_multiple_scenarios(max_p_sub1_list, save_plots=True, output_folder="comparison_plots", dpi=300):
    """
    Run multiple scenarios with different max_p_sub1 values and plot comparison charts
    for power, velocity, substation power, and voltage profiles.
    
    Parameters:
        max_p_sub1_list: List of different max_p_sub1 values to compare
        save_plots: Whether to save the plots to files (default: True)
        output_folder: Folder to save plots in (default: "comparison_plots")
        dpi: Resolution for saved plots (default: 300)
    """
    # Set font sizes to match plot_Pm_and_Pn_profile
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

    # Create output directory if saving is enabled
    if save_plots:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print(f"Plots will be saved to {os.path.abspath(output_folder)}")
    
    # Process speed limits and setup data structure
    speed_limits_dict, speed_limit_array, distance_remaining = process_speed_limits(simulation)
    gradients = collecting_gradients(simulation, mode="const", max_gradient=0.0)
    data = {0: {'grade': gradients[0], 'speed_limit': speed_limit_array[0]}}
    for i in range(1, int(simulation.distance_remaining / simulation.delta_s) + 1):
        data[i * simulation.delta_s] = {
            'grade': gradients[min(i, len(gradients) - 1)],
            'speed_limit': speed_limit_array[min(i, len(speed_limit_array) - 1)]
        }
    
    # Run scenarios with different max_p_sub1 values
    scenario_results = []
    for max_p_sub1_val in max_p_sub1_list:
        # Create a copy of electrical params with different max_p_sub1
        electrical_scenario = ElectricalParams(
            rho=electrical.rho,
            V0=electrical.V0,
            cat_voltage_bound=electrical.cat_voltage_bound,
            Consider_electrical_losses=electrical.Consider_electrical_losses,
            max_p_sub1=max_p_sub1_val,
            max_p_sub2=electrical.max_p_sub2

        )

        # Run the model with these parameters
        print(f"Running scenario with max_p_sub1 = {max_p_sub1_val/1e6} MW...")
        try:
            model, termination_condition = Main(data, train, electrical_scenario, simulation)
            
            if termination_condition in [pyomo.TerminationCondition.optimal, pyomo.TerminationCondition.locallyOptimal]:
                total_energy = calculate_energy_consumption(model, data, simulation, print_results=False)
                scenario_results.append({
                    'max_p_sub1': max_p_sub1_val,
                    'model': model,
                    'energy': total_energy
                })
            else:
                print(f"Scenario max_p_sub1={max_p_sub1_val/1e6} MW: No optimal solution found.")
        except Exception as e:
            print(f"Error in scenario max_p_sub1={max_p_sub1_val/1e6} MW: {str(e)}")

    if not scenario_results:
        print("No valid scenarios to plot.")
        return

    # 1. Plot velocity profiles (distance)
    fig1 = plt.figure(figsize=(12, 6))
    for result in scenario_results:
        model = result['model']
        distances, velocities = [], []
        for d in data.keys():
            distances.append(d / 1000)  # Convert distance to km
            velocities.append(model.v[d]() * 3.6)  # Convert velocity to km/h
        label = f"Substation 1 Power Limit={result['max_p_sub1']/1e6:.2f} MW, Energy Consumption={result['energy']:.3f} kWh"
        plt.plot(distances, velocities, linewidth=2, label=label)
    
    # Add speed limit profile
    interval_starts = list(speed_limits_dict.keys())
    interval_ends = [speed_limits_dict[d]['end'] for d in interval_starts]
    interval_speeds = [speed_limits_dict[d]['speed'] for d in interval_starts]
    plot_distances, plot_speeds = [], []
    for i in range(len(interval_starts)):
        if i == 0:
            plot_distances.append(interval_starts[i] / 1000)
            plot_speeds.append(interval_speeds[i])
        plot_distances.append(interval_ends[i] / 1000)
        plot_speeds.append(interval_speeds[i + 1] if i + 1 < len(interval_speeds) else interval_speeds[i])
    plt.step(plot_distances, plot_speeds, where='post', color='#b58900', linewidth=2, linestyle='--', label='Speed Limit')
    
    plt.xlabel('Distance (km)', fontsize=16, fontweight='bold')
    plt.ylabel('Velocity (km/h)', fontsize=16, fontweight='bold')
    plt.title('Velocity Profile Comparison', fontsize=18, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    if save_plots:
        velocity_plot_path = os.path.join(output_folder, 'velocity_comparison.png')
        plt.savefig(velocity_plot_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved velocity comparison plot to {velocity_plot_path}")
    
    # 2. Plot power profiles (distance)
    fig2 = plt.figure(figsize=(12, 6))
    for result in scenario_results:
        model = result['model']
        distances, powers = [], []
        for d in data.keys():
            distances.append(d / 1000)
            power = (model.Pm[d]()/train.eta - model.Pn[d]()*train.braking_eff) / 1e6  # Convert to MW
            powers.append(power)
        label = f"Substation 1 Power Limit={result['max_p_sub1']/1e6:.2f} MW, Energy Consumption={result['energy']:.3f} kWh"
        plt.plot(distances, powers, linewidth=2, label=label)
    
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.xlabel('Distance (km)', fontsize=16, fontweight='bold')
    plt.ylabel('Power (MW)', fontsize=16, fontweight='bold')
    plt.title('Train Electrical Power Profile Comparison', fontsize=18, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    if save_plots:
        power_plot_path = os.path.join(output_folder, 'power_comparison.png')
        plt.savefig(power_plot_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved power comparison plot to {power_plot_path}")
    
    # 3. Plot substation power profiles (distance)
    fig3 = plt.figure(figsize=(12, 6))
    for result in scenario_results:
        model = result['model']
        distances, p_sub1_values, p_sub2_values = [], [], []
        for d in data.keys():
            if d == 0:
                continue
            distances.append(d / 1000)
            p_sub1 = (simulation.distance_remaining - d) * model.Pg[d]() * 1e-6 / simulation.distance_remaining
            p_sub2 = d * model.Pg[d]() * 1e-6 / simulation.distance_remaining
            p_sub1_values.append(p_sub1)
            p_sub2_values.append(p_sub2)
        
        label1 = f"Substation 1, Limit={result['max_p_sub1']/1e6:.2f} MW"
        label2 = f"Substation 2, Limit={electrical.max_p_sub2/1e6:.2f} MW"
        plt.plot(distances, p_sub1_values, linewidth=2, label=label1)
        plt.plot(distances, p_sub2_values, linewidth=2, linestyle='--', label=label2)
    
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.xlabel('Distance (km)', fontsize=16, fontweight='bold')
    plt.ylabel('Power (MW)', fontsize=16, fontweight='bold')
    plt.title('Substation Power Profile Comparison', fontsize=18, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    if save_plots:
        substation_plot_path = os.path.join(output_folder, 'substation_comparison.png')
        plt.savefig(substation_plot_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved substation power comparison plot to {substation_plot_path}")
    
    # 4. Plot voltage profiles (distance)
    fig4 = plt.figure(figsize=(12, 6))
    for result in scenario_results:
        model = result['model']
        distances, voltages = [], []
        for d in data.keys():
            distances.append(d / 1000)
            voltages.append(model.V[d]())
        label = f"max_p_sub1={result['max_p_sub1']/1e6:.2f} MW"
        plt.plot(distances, voltages, linewidth=2, label=label)
    
    plt.xlabel('Distance (km)', fontsize=16, fontweight='bold')
    plt.ylabel('Voltage (V)', fontsize=16, fontweight='bold')
    plt.title('Voltage Profile Comparison', fontsize=18, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    if save_plots:
        voltage_plot_path = os.path.join(output_folder, 'voltage_comparison.png')
        plt.savefig(voltage_plot_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved voltage comparison plot to {voltage_plot_path}")
    
    # Also save a summary file with the scenario information
    if save_plots:
        summary_path = os.path.join(output_folder, 'scenario_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Scenario Comparison Summary\n")
            f.write(f"=========================\n\n")
            f.write(f"Total Distance: {simulation.distance_remaining/1000:.2f} km\n")
            f.write(f"Maximum Run Time: {simulation.time_remaining/60:.2f} min\n\n")
            f.write(f"Scenarios:\n")
            for result in scenario_results:
                f.write(f"- Substation 1 Power Limit: {result['max_p_sub1']/1e6:.2f} MW, Energy: {result['energy']:.3f} kWh\n")
        print(f"Saved scenario summary to {summary_path}")
    
    plt.show()

if __name__ == "__main__":
    main()