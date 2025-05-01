import pyomo.environ as pyomo
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import numpy as np
import sys
import os

# Electrical Parameters
rho = 0.00003 # Ohms/m
V0 = 1500 # V

# Train Parameters
m = 390000 # kg (train weight, but should be variable later)
A = 3.020*4.670 # m^2 (Frontal area)
C = 0.002 # (Rolling resistance coefficient)
eta = 0.893564 # Efficiency of the train's propulsion system
C_d = 0.8 # (Drag coefficient)
braking_eff = 0.0 # Regenerative braking efficiency    
max_v = 44.444 # m/s = 160 km/h (VIRM)
max_acc = 0.768 # m/s2 (2.76 km/h/s) (VIRM)
WindSpeed = 0 # m/s (Wind speed)

# Distance discretization
S = 10000 # (m) Length between Substation 1 and Substation 2
delta_s = 100  # Distance step in meters
total_time = 400 # (sec) From Substation 1 to Substation 2
max_p_sub1 = 2157000
max_p_sub2 = 2157000
max_p = 2157000 # W (max power)
min_p = max_p # W (min power)



gradients = []
with open('gradients.txt', 'r') as f:
    for line in f:
        gradients.append(float(line.strip()))

data = {0: {'grade': gradients[0]}}
num_steps = int(S / delta_s) + 1  # Number of distance steps
for i in range(1, num_steps):
    distance = i * delta_s
    data[distance] = {'grade': gradients[min(i, len(gradients) - 1)],}

def Initializer(S, delta_s, max_acc, data, m, C_d, A, C, eta):
    D = data.keys()  # Distance steps
    model0 = pyomo.ConcreteModel()
    model0.v = pyomo.Var(D, domain=pyomo.NonNegativeReals, bounds=(0, max_v))  # Velocity (m/s)
    model0.P = pyomo.Var(D, domain=pyomo.Reals)  # Power consumption (kW)
    model0.t = pyomo.Var(D, domain=pyomo.NonNegativeReals)  # Time at each distance step
    model0.of = pyomo.Objective(expr=0.0, sense=pyomo.minimize)
    model0.cons = pyomo.ConstraintList()
    
    final_distance = list(D)[-1]
    # model0.cons.add(model0.s[final_time] == S)
    model0.t[0].fix(0)  # Initial time is 0
    model0.v[0].fix(0)  # Initial velocity is 0
    model0.v[final_distance].fix(0)  # Final velocity is 0
    model0.P[0].fix(0)


    for d in list(D)[1:]:
        prev_d = d - delta_s
        model0.cons.add(model0.t[d] == model0.t[prev_d] + 2 * delta_s / (model0.v[d] + model0.v[prev_d]))
        model0.cons.add((model0.v[d] - model0.v[prev_d]) / (2 * delta_s / (model0.v[d] + model0.v[prev_d])) <= max_acc)
        model0.cons.add(-(model0.v[d] - model0.v[prev_d]) / (2 * delta_s / (model0.v[d] + model0.v[prev_d])) <= max_acc)

        # Simpler version of Davies equation for power consumption
        model0.cons.add(model0.P[d] == 1 / eta * (
            0.5 * 1.225 * C_d * A * model0.v[d]**2 +
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

def train(rho, S, delta_s, max_acc, max_p, data, m, C_d, A, C, eta, braking_eff, total_time):
    v_opt, P_opt = Initializer(S, delta_s, max_acc, data, m, C_d, A, C, eta)
    D = data.keys()
    
    model = pyomo.ConcreteModel()
    
    # Decision Variables
    model.v = pyomo.Var(D, domain=pyomo.NonNegativeReals, bounds=(0, max_v), initialize=lambda model0, d: v_opt[d]) # velocity (m/s)

    # State Variables
    model.Pm = pyomo.Var(D, domain=pyomo.NonNegativeReals, bounds=(0, max_p)) 
    model.Pn = pyomo.Var(D, domain=pyomo.NonNegativeReals, bounds=(0, min_p)) 
    model.P = pyomo.Var(D, domain=pyomo.Reals, initialize=lambda model0, d: P_opt[d])
    model.P_sub1 = pyomo.Var(D, domain=pyomo.Reals, bounds=(-max_p_sub1, max_p_sub1)) 
    model.P_sub2 = pyomo.Var(D, domain=pyomo.Reals, bounds=(-max_p_sub2, max_p_sub2))
    model.t = pyomo.Var(D, domain=pyomo.NonNegativeReals) # Distance 
    
    model.of = pyomo.Objective(expr=sum(model.Pm[d] - model.Pn[d] * braking_eff for d in D), sense=pyomo.minimize)
    
    # Constraints
    model.cons = pyomo.ConstraintList()
    final_distance = list(D)[-1]
        
    # Initial conditions
    model.v[0].fix(0)
    model.v[final_distance].fix(0)
    model.t[0].fix(0)
    model.t[final_distance].fix(total_time)
    # model.cons.add(model.s[final_distance] == S)
    model.P[0].fix(0)
    model.Pm[0].fix(0)
    model.Pn[0].fix(0)
    model.P_sub1[0].fix(0)
    model.P_sub2[0].fix(0)

    for d in list(D)[1:]:
        prev_d = d - delta_s
        model.cons.add(model.t[d] == model.t[prev_d] + 2 * delta_s / (model.v[d] + model.v[prev_d]))
        model.cons.add((model.v[d] - model.v[prev_d]) / (2 * delta_s / (model.v[d] + model.v[prev_d])) <= max_acc)
        model.cons.add(-(model.v[d] - model.v[prev_d]) / (2 * delta_s / (model.v[d] + model.v[prev_d])) <= max_acc)
        model.cons.add(model.P[d] == model.Pm[d] - model.Pn[d])
        model.cons.add(model.P_sub1[d] == model.P[d] * (S - d)/S)
        model.cons.add(model.P_sub2[d] == model.P[d] * d / S)
        
        # Davies equation for power consumption
        model.cons.add(model.P[d] == 1 / eta * (
            0.5 * 1.225 * C_d * A * model.v[d]**2 +
            C * m * 9.807 +
            m * 9.807 * data[d]['grade'] +  # Gradient at this distance
            m * (model.v[d] - model.v[prev_d]) / (2 * delta_s / (model.v[d] + model.v[prev_d]))
        ) * model.v[d]) 
            
    solver = pyomo.SolverFactory('ipopt')
    results = solver.solve(model, tee=True)
     
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
    plt.title(f'Train Velocity Profile (S={S/1000}km)')

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

    plt.title(f'Train Power, Acceleration, and Velocity Profile (S={S/1000} km)')
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
    plt.title(f'Substation Power Profile (S={S/1000} km)')
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
            V = V0 - P / ((V0 / (rho * d + 1e-9)) + (V0 / (rho * (S - d + 1e-9))))
            voltages.append(V)

    plt.figure(figsize=(12, 6))
    plt.plot(distances, voltages, 'b-', linewidth=2)
    plt.xlabel('Distance (km)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Train Voltage Profile (S={S/1000} km)')
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

model, termination_condition = train(rho, S, delta_s, max_acc, max_p, data, m, C_d, A, C, eta, braking_eff, total_time)

if termination_condition in [pyomo.TerminationCondition.optimal, pyomo.TerminationCondition.locallyOptimal]:
    # Save results to file
    base_filename = f"t{total_time}_S{S}"
    ext = ".txt"
    suffix_num = 0
    filename = f"{base_filename}_e{suffix_num}{ext}"
    while os.path.exists(filename):
        suffix_num += 1
        filename = f"{base_filename}_e{suffix_num}{ext}"

    with open(filename, 'w') as f:
        f.write(f"P_sub1_max: {max_p_sub1} W, P_sub2_max: {max_p_sub2} W\n")
        f.write(f"Braking Efficiency: {braking_eff*100}%\n")
        f.write("Distance(m), Velocity(km/h), Power (MW), P_sub1 (MW), P_sub2 (MW)\n")  # Header
        for d in data.keys():
            v_out = 3.6 * model.v[d]()
            if model.P[d]() < 0:
                P_out = braking_eff * model.P[d]() / 1000000  # Added braking_eff for braking power
            else:
                P_out = model.P[d]() / 1000000
            P_sub1_out = model.P_sub1[d]() / 1000000
            P_sub2_out = model.P_sub2[d]() / 1000000
            f.write(f"{d}, {v_out:.3f}, {P_out:.6f}, {P_sub1_out:.6f}, {P_sub2_out:.6f}\n")

    # Plot results
    plot_Pm_and_Pn_profile(model, data)
    plot_Pm_and_Pn_profile_time(model, data)
    plot_substation_powers(model, data)
    plot_voltage_profile(model, data)
    Pm_per_second, total_Pm = calculate_Pm_per_d(model, data)
    plt.show()
else:
    print("No results to save or plot due to solver termination condition.")