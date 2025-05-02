import pyomo.environ as pyomo
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import sys
import os
import random
import time
import numpy as np


# Electrical Parameters
rho = 0.00003 # Ohms/m
V0 = 1500 # V

# Train Parameters (SNG)
Rotatory_inertia_factor = 0.0674
Passenger_weight = 15000 # kg
m = 152743 * (1 + Rotatory_inertia_factor) + Passenger_weight# kg (train weight, but should be variable later)
A = 2.88*4.25 # m^2 (Frontal area)
C = 0.002 # (Rolling resistance coefficient)
eta = 0.857 # Efficiency of the train's propulsion system
C_d = 0.8 # (Drag coefficient)
braking_eff = 0.1 # Regenerative braking efficiency    
max_v = 44.444 # m/s = 160 km/h
max_acc = 0.81 # m/s2 (2.76 km/h/s)
max_braking = 0.5 # m/s2 (2.76 km/h/s)
Auxiliary_power = 65558 #  Average auxiliary power consumption (W)
max_p = 1393000 - Auxiliary_power # W (max power)
# min_p = max_p # W (min power)

# Distance discretization
total_distance = 30000 # (m) Length between Substation 1 and Substation 2
total_time = 15 * 60 # (sec) From Substation 1 to Substation 2

# Time-dependent parameters
max_p_sub1 = 2.0 * 1000000 # W (max power for substation 1)
max_p_sub2 = 2.0 * 1000000 # W (max power for substation 2)
# WindSpeed = random.choice([random.uniform(-5, -2), random.uniform(2, 5)])  # m/s (Wind speed, excluding -2 to 2)
WindSpeed = 2.5

# Initialize initial conditions
v_init = 0 / 3.6 # (Enter in km/h) Initial velocity (m/s)
t_init = 0 * 60 # Initial time (s)
d_init = 0 * 1000 # Initial distance (m)
time_remaining = total_time - t_init # Remaining time (s)
distance_remaining = total_distance - d_init # Remaining distance (m)

# Generate random gradients for the track profile
max_gradient = 0.0015  # Maximum gradient (0.15%)
# gradients = [random.uniform(-max_gradient, max_gradient)]  # Initialize the first gradient randomly within the range
# for i in range(1, distance_remaining // delta_s):
#     prev_gradient = gradients[-1]
#     # Generate a new gradient that does not change more than 0.005 from the previous one
#     new_gradient = prev_gradient + random.uniform(-0.0001, 0.0001)
#     # Ensure the new gradient stays within the range [-0.01, 0.01]
#     new_gradient = max(-max_gradient, min(max_gradient, new_gradient))
#     gradients.append(new_gradient)

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
        model.cons.add(model.P[d] == 1 / eta * (
            0.5 * 1.225 * C_d * A * (model.v[d] + WindSpeed)**2 +
            C * m * 9.807 +
            m * 9.807 * data[d]['grade'] +  # Gradient at this distance
            m * (model.v[d] - model.v[prev_d]) / (2 * delta_s / (model.v[d] + model.v[prev_d]))
        ) * model.v[d]) 
            
    solver = pyomo.SolverFactory('ipopt')
    results = solver.solve(model, tee=False)
    end_train = time.time()
    print(f"Training time: {end_train - start_train:.2f} seconds")

    return model, results.solver.termination_condition  # Add return statement  # Add return statement

def plot_velocity_profile(model, data):
    times = []
    velocities = []
    
    for d in data.keys():
        distances.append(d / 1000)  # Convert to km
        velocities.append(model.v[d]() * 3.6)  # Convert m/s to km/h
    
    plt.figure(figsize=(12, 6))
    plt.plot(distances, velocities, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Distance (km)')
    plt.ylabel('Velocity (km/h)')
    plt.title(f'Train Velocity Profile (S={distance_remaining/1000}km)')

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
    ax1.plot(distances, Pm_values, 'b-', linewidth=2, label='Positive Power (Pm)')
    ax1.plot(distances, Pn_values, 'r-', linewidth=2, label='Negative Power (Pn)')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Power (MW)', color='b')

    # Create secondary y-axis for acceleration
    ax2 = ax1.twinx()
    ax2.plot(distances, accelerations, 'g-', linewidth=2, label='Acceleration')
    ax2.set_ylabel('Acceleration (m/s²)', color='g')

    # Create third y-axis for velocity
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.plot(distances, velocities, 'orange', linewidth=2, label='Velocity')
    ax3.set_ylabel('Velocity (km/h)', color='orange')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')

    plt.title(f'Train Power, Acceleration, and Velocity Profile (S={distance_remaining/1000} km)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)   
    
def plot_substation_powers(model, data):
    distances = []
    P_sub1_values = []
    P_sub2_values = []

    for d in data.keys():
        distances.append(d / 1000)  # Convert distance to km
        P_sub1_values.append(model.P_sub1[d]() / 1000000)  # Convert W to MW
        P_sub2_values.append(model.P_sub2[d]() / 1000000)  # Convert W to MW

    plt.figure(figsize=(12, 6))
    plt.plot(distances, P_sub1_values, 'b-', label='Substation 1 Power')
    plt.plot(distances, P_sub2_values, 'r-', label='Substation 2 Power')
    plt.xlabel('Distance (km)')
    plt.ylabel('Power (MW)')
    plt.title(f'Substation Power Profile (S={distance_remaining/1000} km)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
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

delta_s_list = [2500,2000,1500,1250,1000,750,500,400,300,250,200,150,125,100,75,50]
num_runs = 20  # Number of repetitions for averaging

compile_times_all = {ds: [] for ds in delta_s_list}
energy_consumptions_all = {ds: [] for ds in delta_s_list}

for run in range(num_runs):
    print(f"\nRun {run+1}/{num_runs}")
    for delta_s in delta_s_list:
        print(f"  Running for delta_s = {delta_s} ...")
        num_steps = int(total_distance / delta_s) + 1
        gradients = [max_gradient] * num_steps
        data = {0: {'grade': gradients[0]}}
        for i in range(1, num_steps):
            distance = i * delta_s
            data[distance] = {'grade': gradients[min(i, len(gradients) - 1)]}

        start_time = time.time()
        model, termination_condition = train(
            total_distance, delta_s, max_acc, max_braking, max_p, data, m, C_d, A, C, eta,
            braking_eff, total_time, WindSpeed, v_init, max_p_sub1, max_p_sub2
        )
        end_time = time.time()
        compile_time = end_time - start_time
        if termination_condition in [pyomo.TerminationCondition.optimal, pyomo.TerminationCondition.locallyOptimal]:
            total_energy = calculate_energy_consumption(model, data, delta_s / max_v)
            compile_times_all[delta_s].append(compile_time)
            energy_consumptions_all[delta_s].append(total_energy)
            print(f"    Success: Compile time = {compile_time:.2f} s, Total Energy = {total_energy:.3f} kWh")
        else:
            compile_times_all[delta_s].append(np.nan)
            energy_consumptions_all[delta_s].append(np.nan)
            print(f"    Failed: No optimal solution found.")

# Calculate averages (ignoring failed runs)
avg_compile_times = []
avg_energy_consumptions = []
for ds in delta_s_list:
    times = [t for t in compile_times_all[ds] if not np.isnan(t)]
    energies = [e for e in energy_consumptions_all[ds] if not np.isnan(e)]
    avg_compile_times.append(np.mean(times) if times else np.nan)
    avg_energy_consumptions.append(np.mean(energies) if energies else np.nan)

# Print summary
print("\nAveraged Summary for different delta_s:")
for ds, t, e in zip(delta_s_list, avg_compile_times, avg_energy_consumptions):
    print(f"delta_s={ds:4}: Avg Compile time = {t if not np.isnan(t) else 'N/A':>8.2f}, Avg Total Energy = {e if not np.isnan(e) else 'N/A'}")

# Interactive plot with averages
fig = make_subplots(rows=1, cols=2, subplot_titles=("Avg Compile Time vs delta_s", "Avg Total Energy vs delta_s"))

fig.add_trace(
    go.Scatter(x=delta_s_list, y=avg_compile_times, mode='markers+lines', name='Avg Compile Time (s)', marker=dict(size=10)),
    row=1, col=1
)
fig.update_xaxes(title_text="delta_s (m)", row=1, col=1)
fig.update_yaxes(title_text="Avg Compile Time (s)", row=1, col=1)

fig.add_trace(
    go.Scatter(x=delta_s_list, y=avg_energy_consumptions, mode='markers+lines', name='Avg Total Energy (kWh)', marker=dict(size=10)),
    row=1, col=2
)
fig.update_xaxes(title_text="delta_s (m)", row=1, col=2)
fig.update_yaxes(title_text="Avg Total Energy (kWh)", row=1, col=2)

fig.update_layout(title="Averaged Interactive Comparison of delta_s", hovermode="closest")
fig.show()