import pyomo.environ as pyomo
import matplotlib.pyplot as plt
import sys
import random
import time
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import csv
from dataclasses import dataclass

# === Parameter Classes ===
@dataclass
class ElectricalParams:
    rho: float = 0.00003  # Ohms/m
    V0: int = 1800        # Operational Voltage (V)
    cat_voltage_bound: list = (1200, 1925)  # [min, max] catenary voltage
    Consider_electrical_losses: int = 0     # 0: do not consider, 1: consider
    max_p_sub1: float = 30.5e6       # W
    max_p_sub2: float = 30.0e6       # W
@dataclass
class TrainParams:
    m: float = 391000               # kg
    A: float = 3.02 * 4.67          # m^2
    C: float = 0.002                # Rolling resistance coefficient
    C_d: float = 0.8                # Drag coefficient
    eta: float = 0.87               # Propulsion efficiency
    braking_eff: float = 0.89       # Regenerative braking efficiency
    max_v: float = 44.444           # m/s
    max_acc: float = 0.768          # m/s^2
    max_braking: float = 0.5        # m/s^2
    max_p: float = 359900 * 120       # W
    mu_curve: float = 0.001         # Curve resistance coefficient
@dataclass
class SimulationParams:
    total_time: float = 9.0 * 60    # sec
    delta_s: int = 100              # m
    WindSpeed: float = 0            # m/s
    v_init: float = 0 / 3.6         # m/s
    t_init: float = 0 * 60          # s
    d_init: float = 0 * 1000        # m
    speed_limit_csv_path = r"c:\Users\DarbandiH\OneDrive - University of Twente\Postdoc\Python\Train optimization problem\SpeedLimit_Rta_Gda.csv"

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
        print(f"\nTotal Energy Consumption: {calculate_energy_consumption(model, data, simulation):.3f} kWh")
        plot_substation_powers(model, data, simulation)
        plot_Pm_and_Pn_profile(model, data, simulation, speed_limits=speed_limits_dict)
        # plot_Pm_and_Pn_profile_time(model, data, simulation, speed_limits=speed_limits_dict)
        plot_voltage_profile(model, data, simulation)
        save_power_velocity_acceleration_to_csv("Train_results.csv", model, data, simulation, calculate_energy_consumption(model, data, simulation))
        plt.show()
    else:
        print("No results to save or plot due to solver termination condition.")

    ## Comparison of multiple scenarios, Uncomment the following lines to run the comparison with different max_p_sub1 values
    # max_p_sub1_list = [.5e6, .75e6, 1e6, 2.5e6]
    # plot_multiple_scenarios(max_p_sub1_list)

def collecting_gradients(simulation: SimulationParams, mode, max_gradient): # Generate track gradients profile. Modes: "const" for constant gradient, "randm" for random profile, "files" for extracting from a file.
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
        sys.exit()
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
    model.P = pyomo.Var(D, domain=pyomo.Reals, initialize=lambda model0, d: P_opt[d])
    model.t = pyomo.Var(D, domain=pyomo.NonNegativeReals) # Distance 
    model.V = pyomo.Var(D, domain=pyomo.NonNegativeReals, bounds=(electrical.cat_voltage_bound[0], electrical.cat_voltage_bound[1])) # Voltage (according to ProRail safety standards)

    '''If the objective function is changed, the plot functions and calculate_energy_consumption function must be updated accordingly.'''
    model.of = pyomo.Objective(expr=sum((model.Pm[d]/train.eta - train.braking_eff*model.Pn[d])*(2 * simulation.delta_s / (model.v[d] + model.v[d - simulation.delta_s])) for d in list(D)[1:]), sense=pyomo.minimize)
    
    # Constraints
    model.cons = pyomo.ConstraintList()
    final_distance = list(D)[-1]
    model.v[0].fix(simulation.v_init)
    model.Pm[0].fix(0)
    model.Pn[0].fix(0)
    model.v[final_distance].fix(0)
    model.t[final_distance].fix(simulation.time_remaining)
    
    for d in list(D)[1:]:
        prev_d = d - simulation.delta_s
        model.cons.add(model.t[d] == model.t[prev_d] + 2 * simulation.delta_s / (model.v[d] + model.v[prev_d]+1e-6))
        model.cons.add(model.v[d] <= data[d]['speed_limit'])  # Track speed limit constraint
        model.cons.add((model.v[d] - model.v[prev_d]) / (2 * simulation.delta_s / (model.v[d] + model.v[prev_d]+1e-6)) <= train.max_acc)
        model.cons.add(-(model.v[d] - model.v[prev_d]) / (2 * simulation.delta_s / (model.v[d] + model.v[prev_d]+1e-6)) <= train.max_braking)
        model.cons.add(model.P[d] == model.Pm[d] - model.Pn[d])
        model.cons.add(abs(model.P[d]) <= (simulation.distance_remaining*electrical.max_p_sub1/(simulation.distance_remaining - d + 1e-9)))
        model.cons.add(abs(model.P[d]) <= (simulation.distance_remaining *electrical.max_p_sub2/(d + 1e-9)))
        model.cons.add(model.P[d] == model.V[d] * ((electrical.V0 - model.V[d]) / (electrical.rho * d + 1e-9) + (electrical.V0 - model.V[d]) / (electrical.rho * (simulation.distance_remaining - d) + 1e-9)))  # Electrical power consumption
        model.cons.add(
            model.P[d] == model.v[d] * ( 
                0.5 * 1.225 * train.C_d * train.A * (model.v[d] + simulation.WindSpeed)**2 + # aerodynamic drag
                train.C * train.m * 9.807 + # rolling resistance
                train.m * 9.807 * data[d]['grade'] + # track grade
                train.m * (model.v[d] - model.v[prev_d]) / (2 * simulation.delta_s / (model.v[d] + model.v[prev_d]+1e-6))) # acceleration
            + electrical.rho * simulation.distance_remaining * electrical.Consider_electrical_losses * #electrical losses simplification (can be switched off)
            (model.v[d] / train.eta  / electrical.V0 * (
                0.5 * 1.225 * train.C_d * train.A * (model.v[d] + simulation.WindSpeed)**2 +
                train.C * train.m * 9.807 +
                train.m * 9.807 * data[d]['grade'] +
                train.m * (model.v[d] - model.v[prev_d]) / (2 * simulation.delta_s / (model.v[d] + model.v[prev_d]))
            ))**2)
        
    solver = pyomo.SolverFactory('ipopt')
    results = solver.solve(model, tee=False)
    return model, results.solver.termination_condition  # Add return statement

def plot_Pm_and_Pn_profile_time(model, data, simulation: SimulationParams, speed_limits=None):
    # Initialize with zero at the first point
    times, Pm_values, Pn_values, velocities, accelerations = [0], [0], [0], [0], [0]
    for idx, d in enumerate(data.keys()):
        if idx == 0 or model.t[d]() == 0:
            continue  # Already initialized first point
        t = model.t[d]()
        v = model.v[d]()
        pm = model.Pm[d]() / 1000000  # MW
        pn = model.Pn[d]() / 1000000  # MW
        vel_kmh = v * 3.6
        times.append(t)
        Pm_values.append(pm)
        Pn_values.append(pn)
        velocities.append(vel_kmh)
        prev_d = d - simulation.delta_s
        a = (v - model.v[prev_d]()) / (2 * simulation.delta_s / (v + model.v[prev_d]()))
        accelerations.append(a)

    # Create the plot with two y-axes (power and velocity)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot power on primary y-axis
    ax1.plot(times, [pm/train.eta - train.braking_eff*pn for pm, pn in zip(Pm_values, Pn_values)], 'b-', linewidth=2, label='Power (P)')
    ax1.axhline(0, color='black', linewidth=1, linestyle='-')  # Add horizontal line at y=0
    ax1.axvline(0, color='black', linewidth=2)  # y-axis at x=0
    ax1.set_xlim(left=0)
    ax1.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Power (MW)', color='b', fontsize=16, fontweight='bold')

    # Add more grid lines for y-axis only
    ax1.yaxis.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.xaxis.grid(False)

    # Add more ticks for y axis
    y_major_locator = MultipleLocator(0.2)  # Major ticks every 0.2 MW
    y_minor_locator = AutoMinorLocator(4)   # 4 minor ticks between majors
    ax1.yaxis.set_major_locator(y_major_locator)
    ax1.yaxis.set_minor_locator(y_minor_locator)

    # Create second y-axis for velocity (move from third to second axis)
    ax2 = ax1.twinx()
    ax2.plot(times, velocities, 'orange', linewidth=2, label='Velocity')
    ax2.set_ylabel('Velocity (km/h)', color='orange', fontsize=16, fontweight='bold')

    # Overlay speed limit profile from CSV if provided
    if speed_limits is not None:
        interval_starts = list(speed_limits.keys())
        interval_ends = [speed_limits[d]['end'] for d in interval_starts]
        interval_speeds = [speed_limits[d]['speed'] for d in interval_starts]
        plot_times,plot_speeds = [],[]
        for i in range(len(interval_starts)):
            d_start , d_end = interval_starts[i] , interval_ends[i]
            d_keys = list(data.keys())
            t_start = model.t[d_keys[min(range(len(d_keys)), key=lambda j: abs(d_keys[j] - d_start))]]()
            t_end = model.t[d_keys[min(range(len(d_keys)), key=lambda j: abs(d_keys[j] - d_end))]]()
            if i == 0:
                plot_times.append(t_start)
                plot_speeds.append(interval_speeds[i])
            plot_times.append(t_end)
            plot_speeds.append(interval_speeds[i + 1] if i + 1 < len(interval_speeds) else interval_speeds[i])
        # MRSP in a dark yellow color
        ax2.step(plot_times, plot_speeds, where='post', color='#b58900', linewidth=2, linestyle='--', label='MRSP')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    minutes, seconds = int(simulation.time_remaining // 60), int(simulation.time_remaining % 60)
    total_energy = calculate_energy_consumption(model, data, simulation)
    plt.title(
        f'Train Power and Velocity Profile (S={simulation.distance_remaining/1000} km, Run time={minutes} min {seconds} sec, Energy={total_energy:.3f} kWh)',
        fontsize=18, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)   

def plot_Pm_and_Pn_profile(model, data, simulation: SimulationParams, speed_limits=None):
    # Initialize with zero at the first point
    distances, Pm_values, Pn_values, velocities, accelerations = [0], [0], [0], [0], [0]

    for idx, d in enumerate(data.keys()):
        if idx == 0 or d == 0:
            continue  # Already initialized first point
        v = model.v[d]()
        pm = model.Pm[d]() / 1000000  # MW
        pn = model.Pn[d]() / 1000000  # MW
        vel_kmh = v * 3.6
        distances.append(d/1000)
        Pm_values.append(pm)
        Pn_values.append(pn)
        velocities.append(vel_kmh)
        prev_d = d - simulation.delta_s
        a = (v - model.v[prev_d]()) / (2 * simulation.delta_s / (v + model.v[prev_d]()))
        accelerations.append(a)

    # Create the plot with two y-axes (power and velocity)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot power on primary y-axis
    ax1.plot(distances, [pm/train.eta - train.braking_eff*pn for pm, pn in zip(Pm_values, Pn_values)], 'b-', linewidth=2, label='Power (P)')
    ax1.axhline(0, color='black', linewidth=1, linestyle='-')  # Add horizontal line at y=0
    ax1.axvline(0, color='black', linewidth=2)  # y-axis at x=0
    ax1.set_xlim(left=0)
    ax1.set_xlabel('Distance (km)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Power (MW)', color='b', fontsize=16, fontweight='bold')

    # Add more grid lines for y-axis only
    ax1.yaxis.grid(True, which='major', linestyle='--', alpha=0.5)
    ax1.xaxis.grid(True, which='major', linestyle='--', alpha=0.5)

    # Add more ticks for y axis
    y_major_locator = MultipleLocator(0.25)  # Major ticks every 0.25 MW
    # y_minor_locator = AutoMinorLocator(4)   # 4 minor ticks between majors
    ax1.yaxis.set_major_locator(y_major_locator)
    # ax1.yaxis.set_minor_locator(y_minor_locator)

    # Create second y-axis for velocity (move from third to second axis)
    ax2 = ax1.twinx()
    ax2.plot(distances, velocities, 'orange', linewidth=2, label='Velocity')
    ax2.set_ylabel('Velocity (km/h)', color='orange', fontsize=16, fontweight='bold')

    # Overlay speed limit profile from CSV if provided
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
        # MRSP in a dark yellow color
        ax2.step(plot_distances, plot_speeds, where='post', color='#b58900', linewidth=2, linestyle='--', label='MRSP')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    minutes = int(simulation.time_remaining // 60)
    seconds = int(simulation.time_remaining % 60)
    total_energy = calculate_energy_consumption(model, data, simulation)
    plt.title(
        f'Train Power and Velocity Profile (S={simulation.distance_remaining/1000} km, Run time={minutes} min {seconds} sec, Energy={total_energy:.3f} kWh)',
        fontsize=18, fontweight='bold')
    # plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
def plot_substation_powers(model, data, simulation: SimulationParams):
    distances = []
    P_sub1_values = []
    P_sub2_values = []

    for d in data.keys():
        distances.append(d / 1000)  # Convert distance to km
        pm = model.Pm[d]()
        pn = model.Pn[d]()
        P = pm - train.braking_eff * pn
        P_sub1 = (simulation.distance_remaining - d) * P / simulation.distance_remaining
        P_sub2 = d * P / simulation.distance_remaining
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
    ax1.yaxis.grid(True, which='major', linestyle='--', alpha=0.5)
    ax1.xaxis.grid(True)

    # Add more ticks for y axis
    y_major_locator = MultipleLocator(0.25)  # Major ticks every 0.2 MW
    # y_minor_locator = AutoMinorLocator(4)   # 4 minor ticks between majors
    ax1.yaxis.set_major_locator(y_major_locator)
    # ax1.yaxis.set_minor_locator(y_minor_locator)

    minutes, seconds = int(simulation.time_remaining // 60), int(simulation.time_remaining % 60)
    plt.title(f'Substation Power Profile (S={simulation.distance_remaining/1000} km, Run time={minutes} min {seconds} sec)', fontsize=18, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=14)
    
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

def calculate_energy_consumption(model, data, simulation: SimulationParams):
    """Calculates total energy consumption exactly as in the objective function:
    sum of (Pm - Pn * braking_eff) * delta_t for each segment, in kWh."""
    total_energy = 0  # kWh
    for d in list(data.keys())[1:]:
        prev_d = d - simulation.delta_s
        # Average delta_t for the segment
        v1 = model.v[d]()
        v0 = model.v[prev_d]()
        if v1 + v0 > 0:
            delta_t = 2 * simulation.delta_s / (v1 + v0)
        else:
            delta_t = 0
        # Average Pm and Pn for the segment
        Pm_avg = (model.Pm[d]() + model.Pm[prev_d]()) / 2
        Pn_avg = (model.Pn[d]() + model.Pn[prev_d]()) / 2
        # Energy for this segment (W * s = J), convert to kWh
        segment_energy = (Pm_avg/train.eta - Pn_avg * train.braking_eff) * delta_t / 3.6e6
        total_energy += segment_energy
    return total_energy

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
    print(f"Start distance: {start_distance} m, End distance: {end_distance} m")

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
            power_mw = (model.Pm[d]()-model.Pn[d]()*train.braking_eff) / 1e6  # Convert power to MW
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

def plot_multiple_scenarios(max_p_sub1_list, speed_limits_dict=None, data=None, train=None, electrical=None, simulation=None):
    """
    Run multiple scenarios with different max_p_sub1 values and plot velocity and power profiles for comparison.
    Each scenario's energy is shown in the legend.
    MSRP (speed limit) is shown on velocity plots.
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

    scenario_results = []
    for max_p_sub1_val in max_p_sub1_list:
        # Copy simulation object and set max_p_sub1 for this scenario
        sim_scenario = SimulationParams(
            total_time=simulation.total_time,
            delta_s=simulation.delta_s,
            max_p_sub1=max_p_sub1_val,
            max_p_sub2=electrical.max_p_sub2,
            WindSpeed=simulation.WindSpeed,
            v_init=simulation.v_init,
            t_init=simulation.t_init,
            d_init=simulation.d_init
        )
        sim_scenario.time_remaining = sim_scenario.total_time - sim_scenario.t_init
        sim_scenario.distance_remaining = simulation.distance_remaining

        model, termination_condition = Main(
            sim_scenario.distance_remaining, sim_scenario.delta_s, data, train, electrical, sim_scenario
        )
        if termination_condition in [pyomo.TerminationCondition.optimal, pyomo.TerminationCondition.locallyOptimal]:
            total_energy = calculate_energy_consumption(model, data, simulation)
            scenario_results.append({
                'max_p_sub1': max_p_sub1_val,
                'model': model,
                'energy': total_energy,
                'sim': sim_scenario
            })
        else:
            print(f"Scenario max_p_sub1={max_p_sub1_val}: No results due to solver termination condition.")

    # Plot velocity profiles (time)
    plt.figure(figsize=(12, 6))
    for result in scenario_results:
        model = result['model']
        sim = result['sim']
        times = []
        velocities = []
        for idx, d in enumerate(data.keys()):
            t = model.t[d]()
            v = model.v[d]()
            vel_kmh = v * 3.6
            if idx == 0 or t == 0:
                times.append(0)
                velocities.append(0)
            else:
                times.append(t)
                velocities.append(vel_kmh)
        label = f"max_p_sub1={result['max_p_sub1']/1e6:.2f} MW, Energy={result['energy']:.3f} kWh"
        plt.plot(times, velocities, linewidth=2, label=label)
    # Add MSRP (speed limit) profile to velocity vs time plot
    if speed_limits_dict is not None and scenario_results:
        interval_starts = list(speed_limits_dict.keys())
        interval_ends = [speed_limits[d]['end'] for d in interval_starts]
        interval_speeds = [speed_limits[d]['speed'] for d in interval_starts]
        plot_times = []
        plot_speeds = []
        # Use the first model for time mapping
        model = scenario_results[0]['model']
        for i in range(len(interval_starts)):
            d_start = interval_starts[i]
            d_end = interval_ends[i]
            d_keys = list(data.keys())
            t_start = model.t[d_keys[min(range(len(d_keys)), key=lambda j: abs(d_keys[j] - d_start))]]()
            t_end = model.t[d_keys[min(range(len(d_keys)), key=lambda j: abs(d_keys[j] - d_end))]]()
            if i == 0:
                plot_times.append(t_start)
                plot_speeds.append(interval_speeds[i])
            plot_times.append(t_end)
            plot_speeds.append(interval_speeds[i + 1] if i + 1 < len(interval_speeds) else interval_speeds[i])
        plt.step(plot_times, plot_speeds, where='post', color='#b58900', linewidth=2, linestyle='--', label='MRSP')
    plt.xlabel('Time (s)', fontsize=16, fontweight='bold')
    plt.ylabel('Velocity (km/h)', fontsize=16, fontweight='bold')
    plt.title('Velocity Profile Comparison')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # Plot power profiles (time)
    plt.figure(figsize=(12, 6))
    for result in scenario_results:
        model = result['model']
        times = []
        powers = []
        for idx, d in enumerate(data.keys()):
            t = model.t[d]()
            pm = model.Pm[d]()
            pn = model.Pn[d]()
            power = (pm - train.braking_eff * pn) / 1e6  # MW
            if idx == 0 or t == 0:
                times.append(0)
                powers.append(0)
            else:
                times.append(t)
                powers.append(power)
        label = f"max_p_sub1={result['max_p_sub1']/1e6:.2f} MW, Energy={result['energy']:.3f} kWh"
        plt.plot(times, powers, linewidth=2, label=label)
    plt.xlabel('Time (s)', fontsize=16, fontweight='bold')
    plt.ylabel('Power (MW)', fontsize=16, fontweight='bold')
    plt.title('Power Profile Comparison')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # Plot velocity profiles (distance)
    plt.figure(figsize=(12, 6))
    for result in scenario_results:
        model = result['model']
        distances = []
        velocities = []
        for idx, d in enumerate(data.keys()):
            v = model.v[d]()
            vel_kmh = v * 3.6
            if idx == 0 or d == 0:
                distances.append(0)
                velocities.append(0)
            else:
                distances.append(d/1000)
                velocities.append(vel_kmh)
        label = f"max_p_sub1={result['max_p_sub1']/1e6:.2f} MW, Energy={result['energy']:.3f} kWh"
        plt.plot(distances, velocities, linewidth=2, label=label)
    # Add MSRP (speed limit) profile to velocity vs distance plot
    if speed_limits_dict is not None:
        interval_starts = list(speed_limits_dict.keys())
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
        plt.step(plot_distances, plot_speeds, where='post', color='#b58900', linewidth=2, linestyle='--', label='MRSP')
    plt.xlabel('Distance (km)', fontsize=16, fontweight='bold')
    plt.ylabel('Velocity (km/h)', fontsize=16, fontweight='bold')
    plt.title('Velocity Profile Comparison (Distance)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # Plot power profiles (distance)
    plt.figure(figsize=(12, 6))
    for result in scenario_results:
        model = result['model']
        distances = []
        powers = []
        for idx, d in enumerate(data.keys()):
            pm = model.Pm[d]()
            pn = model.Pn[d]()
            power = (pm - train.braking_eff * pn) / 1e6  # MW
            if idx == 0 or d == 0:
                distances.append(0)
                powers.append(0)
            else:
                distances.append(d/1000)
                powers.append(power)
        label = f"max_p_sub1={result['max_p_sub1']/1e6:.2f} MW, Energy={result['energy']:.3f} kWh"
        plt.plot(distances, powers, linewidth=2, label=label)
    plt.xlabel('Distance (km)', fontsize=16, fontweight='bold')
    plt.ylabel('Power (MW)', fontsize=16, fontweight='bold')
    plt.title('Power Profile Comparison (Distance)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()