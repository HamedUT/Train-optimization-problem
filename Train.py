import pyomo.environ as pyomo
import matplotlib.pyplot as plt
import sys
import os
import random
import time
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# Electrical Parameters
rho, V0 = 0.00003, 1500 # Ohms/m, Voltage (V)

# Train Parameters (SNG)
Rotatory_inertia_factor = 0.0674
m = 152743 * (1 + Rotatory_inertia_factor) # kg (train weight, but should be variable later)
A, C, C_d = 2.88*4.25, 0.002, 0.8 # m^2 (Frontal area), (Rolling resistance coefficient), (Drag coefficient)
eta, braking_eff = 0.857, 0.1 # Efficiency of the train's propulsion system, Regenerative braking efficiency
max_v, max_acc, max_braking = 44.444, 0.81, 0.5 # m/s = 160 km/h (max velocity), m/s2 (max acceleration), (max braking)
Auxiliary_power = 65558 #  Average auxiliary power consumption (W)
max_p = 1393000 - Auxiliary_power # W (max power)

# Distance discretization
total_distance = 30000 # (m) Length between Substation 1 and Substation 2
total_time = 15 * 60 # (sec) From Substation 1 to Substation 2
delta_s = 250  # Distance step in meters

# Time-dependent parameters
max_p_sub1 = 2.0 * 1000000 # W (max power for substation 1)
max_p_sub2 = 2.0 * 1000000 # W (max power for substation 2)
# WindSpeed = random.choice([random.uniform(-5, -2), random.uniform(2, 5)])  # m/s (Wind speed, excluding -2 to 2)
WindSpeed = 2.5

# Initialize initial conditions
v_init = 0 / 3.6 # (Enter in km/h) Initial velocity (m/s)
t_init = 0 * 60 # Initial time (s)
d_init = 0 * 1000 # Initial distance (m)
time_remaining, distance_remaining = total_time - t_init, total_distance - d_init # Remaining time (s), Remaining distance (m)

def generate_gradients(distance_remaining, delta_s, mode, max_gradient):
    # Generate gradients for the track profile. mode: "const" for constant gradient, "randm" for random profile.
    num_steps = distance_remaining // delta_s + 1
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

gradients = generate_gradients(distance_remaining, delta_s, mode="const", max_gradient=0.0015)
data = {0: {'grade': gradients[0]}}
for i in range(1, int(distance_remaining / delta_s) + 1): # Number of distance steps
    data[i * delta_s] = {'grade': gradients[min(i, len(gradients) - 1)],}

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

def train(distance_remaining, delta_s, max_acc, max_braking, max_p, data, m, C_d, A, C, eta, braking_eff, time_remaining, WindSpeed, v_init, max_p_sub1, max_p_sub2):
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
        model.cons.add((model.v[d] - model.v[prev_d]) / (2 * delta_s / (model.v[d] + model.v[prev_d])) <= max_acc)
        model.cons.add(-(model.v[d] - model.v[prev_d]) / (2 * delta_s / (model.v[d] + model.v[prev_d])) <= max_braking)        
        model.cons.add(model.P[d] == model.Pm[d] - model.Pn[d])
        model.cons.add(abs(model.P[d])<= distance_remaining*max_p_sub1/(distance_remaining - d + 1e-9))  # Avoid division by zero
        model.cons.add(abs(model.P[d])<= distance_remaining *max_p_sub2/(d + 1e-9))  # Avoid division by zero
        
        # Davies equation for power consumption
        model.cons.add(model.P[d] == model.v[d] / eta * (
            0.5 * 1.225 * C_d * A * (model.v[d] + WindSpeed)**2 +
            C * m * 9.807 +
            m * 9.807 * data[d]['grade'] +  # Gradient at this distance
            m * (model.v[d] - model.v[prev_d]) / (2 * delta_s / (model.v[d] + model.v[prev_d])))) 
        
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

def plot_Pm_and_Pn_profile(model, data):
    distances = []
    Pm_values = []
    Pn_values = []
    accelerations = []
    velocities = []

    for d in data.keys():
        distances.append(d / 1000)  # Convert distance to km
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
    plt.show()
    
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

def calculate_energy_consumption(model, data, delta_t):
    total_energy = 0  # Initialize total energy in joules
    for d in data.keys():
        total_energy += (model.P[d]() * delta_t) / 3.6e6 
    return total_energy

model, termination_condition = train(distance_remaining, delta_s, max_acc, max_braking, max_p, data, m, C_d, A, C, eta, braking_eff, time_remaining, WindSpeed, v_init, max_p_sub1, max_p_sub2)
end_time = time.time()
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
    plot_Pm_and_Pn_profile(model, data)
    # plot_Pm_and_Pn_profile_time(model, data)
    plot_substation_powers(model, data)
    # plot_voltage_profile(model, data)
    # plot_distance_vs_time(model, data)  # Call the new function here
    # plot_gradients_vs_distance(data, S, delta_s)  # Call the new function here
    # Pm_per_second, total_Pm = calculate_Pm_per_d(model, data)
    print(f"\nTotal Energy Consumption: {calculate_energy_consumption(model, data, delta_s / max_v):.3f} kWh")
    # print(f"\nWind Speed: {WindSpeed:0.2f} m/s")
    # print(f"\nTime spent compiling: {end_time - start_time:.2f} seconds")

    plt.show()
else:
    print("No results to save or plot due to solver termination condition.")