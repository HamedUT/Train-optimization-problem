# v3: Added Pm and Pn so the model decides for itself how much power to put
# v4: Added regenrative braking option in the objective function. Also calculated P of substations after model is solved.
# v5: Added P of substations as variables in the model. Also added braking_eff when writing to the txt file for further plot comparison. because when braking, all of the power is not going back to the grid
# v6: Added Voltage (After the solver and not in the model)
# v7: Adding the option to limit in terms of time instead of a constant value using the already created "data" dictionary. It reads data from a txt file and then set the limits on the train function only and not on the initializer.
# v7_MinRunTime: Finding the minimum run time with 10 seconds precision (conditions: max power and max speed in all timesteps) and then running the model with that time plus a margin of safety (10 seconds).

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
max_p = 2157000 # W (max power)
min_p = max_p # W (min power)
eta = 0.893564 # Efficiency of the train's propulsion system
C_d = 0.8 # (Drag coefficient)
braking_eff = 0.0 # Regenerative braking efficiency    
max_v = 44.444 # m/s = 160 km/h (VIRM)
max_acc = 0.768 # m/s2 (2.76 km/h/s) (VIRM)

# Enviromental Variables
S = 10000 # (m) Length between Substation 1 and Substation 2
delta_t = 1 # Seconds
theta = 0.004 # (Gradient)
WindSpeed = 0 # m/s (Wind speed)


#Search variables
Margin = 105 # Safety margin in percentage
min_time = 360 #minimum limit for search
max_time = 391 #maximum limit for search
#substation 1
limited_time_start_1 = 200 # Time to start the restriction
limited_time_end_1 = 300 # Time to end the restriction
limited_power_1 = max_p * 0.5  # Set power limit to 50% of max_p during restriction period
#substation 2
limited_time_start_2 = 200 # Time to start the restriction
limited_time_end_2 = 300 # Time to end the restriction
limited_power_2 = max_p * 0.5  # Set power limit to 50% of max_p during restriction period

def Initializer(S, delta_t, max_acc, data, m, C_d, A, C, theta, eta):
    T = data.keys()
    model0 = pyomo.ConcreteModel()
    model0.v = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0,max_v)) # velocity (m/s)
    model0.P = pyomo.Var(T, domain=pyomo.Reals) # Power consumption (kW)
    model0.s = pyomo.Var(T, domain=pyomo.NonNegativeReals) # Distance
    model0.of = pyomo.Objective(expr = 0.0, sense=pyomo.minimize)
    model0.cons = pyomo.ConstraintList()
    final_time = list(T)[-1]
    model0.cons.add(model0.s[final_time] == S)
    model0.v['00:00'].fix(0)
    model0.v[final_time].fix(0)
    model0.s['00:00'].fix(0)
    model0.P['00:00'].fix(0)

    for t in list(T)[1:]:
        model0.cons.add(model0.v[t] == (model0.s[t] - model0.s[data[t]['t_prev']]) / delta_t)
        model0.cons.add((model0.v[t] - model0.v[data[t]['t_prev']])/delta_t <= max_acc)
        model0.cons.add(-max_acc <= (model0.v[t] - model0.v[data[t]['t_prev']])/delta_t)

        # Simpler version of Davies equation for power consumption
        model0.cons.add(model0.P[t] == 1/eta * (
            0.5 * 1.225 * C_d * A * model0.v[t]**2 +
            C * m * 9.807 +
            m * 9.807 * theta +
            m * (model0.v[t] - model0.v[data[t]['t_prev']])/delta_t
            )* model0.v[t])    
    solver = pyomo.SolverFactory('ipopt')
    # solver.options['max_iter'] = 15000
    # solver.options['tol'] = 1e-5
    # solver.options['acceptable_tol'] = 1e-5
    # solver.options['mu_init'] = 1e-1

    results = solver.solve(model0, tee=False)
    if results.solver.termination_condition == pyomo.TerminationCondition.optimal:
        v_opt = {t: model0.v[t].value for t in T}
        P_opt = {t: model0.P[t].value for t in T}
    else:
        print(f"Infeasible during Initialization - {results.solver.termination_condition}")
        sys.exit()
    return v_opt, P_opt

def train(rho, S, delta_t, max_acc, max_p, data, m, C_d, A, C, theta, eta, braking_eff):

    v_opt, P_opt = Initializer(S, delta_t, max_acc, data, m, C_d, A, C, theta, eta)
    T = data.keys()
    
    model = pyomo.ConcreteModel()
    
    def velocity_bounds(model, t):
        return (0, data[t]['v_max'])
    def power_bounds_1(model, t):
        return (-data[t]['P_sub1_max'], data[t]['P_sub1_max'])
    def power_bounds_2(model, t):
        return (-data[t]['P_sub2_max'], data[t]['P_sub2_max'])
    # Decision Variables
    model.v = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=velocity_bounds, initialize=lambda model0, t: v_opt[t]) # velocity (m/s)

    # State Variables
    model.Pm = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0, max_p)) 
    model.Pn = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0, min_p)) 
    model.P = pyomo.Var(T, domain=pyomo.Reals, initialize=lambda model0, t: P_opt[t])
    model.P_sub1 = pyomo.Var(T, domain=pyomo.Reals, bounds=power_bounds_1) 
    model.P_sub2 = pyomo.Var(T, domain=pyomo.Reals, bounds=power_bounds_2)
    model.s = pyomo.Var(T, domain=pyomo.NonNegativeReals) # Distance 
    # model.d = pyomo.Var(T, domain=pyomo.Binary)
    # model.V = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0, V0-1)) # Voltage
    #model.R1 = pyomo.Var(T, domain=pyomo.NonNegativeReals) # Resistance branch 1
    #model.R2 = pyomo.Var(T, domain=pyomo.NonNegativeReals) # Resistance branch 2

    # model.of = pyomo.Objective(expr = sum(pyomo.Expr_if(model.P[t] >= 0, model.P[t]**2, (-braking_eff*(model.P[t]))**2) for t in T))
    model.of = pyomo.Objective(expr = sum(model.Pm[t] - model.Pn[t]*braking_eff for t in T), sense=pyomo.minimize)
    
    # Constraints
    model.cons = pyomo.ConstraintList()
    final_time = list(T)[-1]
        
    # Initial conditions
    model.v['00:00'].fix(0)
    model.v[final_time].fix(0)
    model.s['00:00'].fix(0)
    model.cons.add(model.s[final_time] == S)
    model.P['00:00'].fix(0)
    model.Pm['00:00'].fix(0)
    model.Pn['00:00'].fix(0)
    model.P_sub1['00:00'].fix(0)
    model.P_sub2['00:00'].fix(0)

    for t in list(T)[1:]:
        model.cons.add(model.v[t] == (model.s[t] - model.s[data[t]['t_prev']]) / delta_t)
        model.cons.add((model.v[t] - model.v[data[t]['t_prev']])/delta_t <= max_acc)
        model.cons.add(-max_acc <= (model.v[t] - model.v[data[t]['t_prev']])/delta_t)
        model.cons.add(model.P[t] == model.Pm[t] - model.Pn[t])
        model.cons.add(model.P_sub1[t] == model.P[t] * (S - model.s[t])/S)
        model.cons.add(model.P_sub2[t] == model.P[t] * model.s[t] / S)

        # Davies equation for power consumption
        # model.cons.add(model.P[t] == (1/eta) * (
        #     0.5 * rho * C_d * A * model.v[t]**2 + #Aerodynamic resistance
        #     C * m * g +
        #     m * g * theta +
        #     Wind Resistance model according to Alessio's Paper
        #     Cr* m * g * model.v[t] + #Curve resistance
        #     m * (model.v[t] - model.v[data[t]['t_prev']]) / delta_t) * model.v[t])      
        
        # Simpler version of Davies equation for power consumption
        model.cons.add(model.P[t] == 1/eta * (
            (0.5 * 1.225 * C_d * A * model.v[t]**2 +
            C * m * 9.807 +                         
            m * 9.807 * theta +                     
            m * (model.v[t] - model.v[data[t]['t_prev']])/delta_t) 
            * model.v[t]))   
            
    solver = pyomo.SolverFactory('ipopt')
    results = solver.solve(model, tee=False)
    return model, results.solver.termination_condition  # Add return statement  # Add return statement

def create_power_limits(test_time, max_p, limited_power, limited_time_start, limited_time_end):
    """Create power limits with restriction during specific period"""
    power_limits = []
    for i in range(test_time):
        if limited_time_start <= i <= limited_time_end:
            power_limits.append(limited_power)  # Restricted power during specified period
        else:
            power_limits.append(max_p)  # Normal maximum power otherwise
    return power_limits

def find_minimum_time(S, delta_t, max_acc, max_p, max_v, m, C_d, A, C, theta, eta, braking_eff, min_time, max_time, limited_power_1, limited_time_start_1, limited_time_end_1, limited_power_2, limited_time_start_2, limited_time_end_2):
    optimal_time = None
    
    while max_time - min_time > 10:  # 10-second precision
        test_time = (min_time + max_time) // 2
        print(f"Testing time: {test_time} seconds...")
        
        # Create data dictionary for this time
        speed_limits = [max_v] * test_time
        power_limits_1 = create_power_limits(test_time, max_p, limited_power_1, limited_time_start_1, limited_time_end_1)
        power_limits_2 = create_power_limits(test_time, max_p, limited_power_2, limited_time_start_2, limited_time_end_2)
        
        test_data = {'00:00': {'v_max': speed_limits[0], 't_prev': '', 'P_sub1_max': power_limits_1[0], 'P_sub2_max': power_limits_2[0]},}
        for i in range(1, test_time):
            minutes = i // 60
            seconds = i % 60
            time = f'{minutes:02d}:{seconds:02d}'
            prev_minutes = (i-1) // 60 
            prev_seconds = (i-1) % 60
            prev_time = f'{prev_minutes:02d}:{prev_seconds:02d}'
            test_data[time] = {'v_max': speed_limits[i], 't_prev': prev_time, 'P_sub1_max': power_limits_2[i], 'P_sub2_max': power_limits_2[i]}
        
        try:
            model, term_cond = train(rho, S, delta_t, max_acc, max_p, test_data, m, C_d, A, C, theta, eta, braking_eff)
            
            if term_cond == pyomo.TerminationCondition.optimal:
                optimal_time = test_time
                max_time = test_time  # Try a lower time
                print(f"Time {test_time}s is feasible. Trying lower time...")
            else:
                min_time = test_time  # Try a higher time
                print(f"Time {test_time}s is infeasible. Trying higher time...")
        except:
            min_time = test_time  # If error occurs, try higher time
            print(f"Error at time {test_time}s. Trying higher time...")
    
    return optimal_time

def plot_velocity_profile(model, data):
    # Extract time and velocity data
    times = []
    velocities = []
    
    for t in data.keys():
        # Convert MM:SS to seconds for x-axis
        minutes, seconds = map(int, t.split(':'))
        times.append(minutes*60 + seconds)
        # Convert velocity to km/h
        velocities.append(model.v[t]())
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, velocities, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (km/h)')
    plt.title(f'Train Velocity Profile (S={S/1000}km, O.F.={(model.of()**0.5)/3600000:.3f} kWh)')
    
    # Add horizontal and vertical gridlines
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Save the plot
    # plt.savefig(f'velocity_profile_S{S}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_Pm_and_Pn_profile(model, data):
    # Extract time and data
    times = []
    Pm_values = []
    Pn_values = []
    accelerations = []
    velocities = []
    # P_actual_values = []
    
    for t in data.keys():
        # Convert MM:SS to seconds for x-axis
        minutes, seconds = map(int, t.split(':'))
        times.append(minutes*60 + seconds)
        # Get power values in MW
        Pm_values.append(model.Pm[t]()/1000000)  # Convert W to MW
        Pn_values.append(model.Pn[t]()/1000000)  # Convert W to MW
        # Get velocity in m/s
        v = model.v[t]()
        velocities.append(v*3.6)  # Convert m/s to km/h
        # Calculate acceleration
        if t == '00:00':
            a = 0
        else:
            a = (model.v[t]() - model.v[data[t]['t_prev']]())/delta_t
        accelerations.append(a)
        
        # Calculate actual power
        # P_actual = abs(1/0.893564 * (
        #     0.5 * 1.225 * 0.8 * 3.020*4.670 * v**2 + 
        #     0.002 * 390000 * 9.807 + 
        #     390000 * 9.807 * 0.004 + 
        #     390000 * a
        # ) * v)
        # P_actual_values.append(P_actual/1000000)  # Convert W to MW
    
    # Create the plot with three y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot power on primary y-axis
    ax1.plot(times, Pm_values, 'b-', linewidth=2, label='Positive Power (Pm)')
    ax1.plot(times, Pn_values, 'r-', linewidth=2, label='Negative Power (Pn)')
    # ax1.plot(times, P_actual_values, 'm--', linewidth=2, label='Actual Power')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Power (MW)', color='b')
    
    # Create secondary y-axis for acceleration
    ax2 = ax1.twinx()
    ax2.plot(times, accelerations, 'g-', linewidth=2, label='Acceleration')
    ax2.set_ylabel('Acceleration (m/s²)', color='g')
    
    # Create third y-axis for velocity
    ax3 = ax1.twinx()
    # Offset the third y-axis to the right
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.plot(times, velocities, 'orange', linewidth=2, label='Velocity')
    ax3.set_ylabel('Velocity (km/h)', color='orange')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')
    
    plt.title(f'Train Power, Acceleration and Velocity Profile (S={S/1000}km)')
    
    # Add more horizontal grid lines
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax3.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.xaxis.grid(True, linestyle='--', alpha=0.7)
    
def plot_substation_powers(model, data):
    times = []
    P_sub1_values = []
    P_sub2_values = []
    distances = []
    
    for t in data.keys():
        minutes, seconds = map(int, t.split(':'))
        times.append(minutes*60 + seconds)
        P_sub1_values.append(model.P_sub1[t]()/1000000)  # Convert to MW
        P_sub2_values.append(model.P_sub2[t]()/1000000)  # Convert to MW
        distances.append(model.s[t]()/1000)  # Convert to km
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot power vs time
    ax1.plot(times, P_sub1_values, 'b-', label='Substation 1')
    ax1.plot(times, P_sub2_values, 'r-', label='Substation 2')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Power (MW)')
    plt.title(f'Substations Power Profile (S={S/1000}km, Braking Efficiency={braking_eff:.0f})')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot power vs distance
    ax2.plot(distances, P_sub1_values, 'b-', label='Substation 1')
    ax2.plot(distances, P_sub2_values, 'r-', label='Substation 2')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Power (MW)')
    plt.title(f'Substations Power Profile (S={S/1000}km, Braking Efficiency={braking_eff:.0f})')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()

def plot_voltage_profile(model, data):
    # Extract time and voltage data
    times = []
    voltages = []
    distances = []
    
    for t in data.keys():
        # Convert MM:SS to seconds for x-axis
        minutes, seconds = map(int, t.split(':'))
        times.append(minutes*60 + seconds)
        
        # Calculate voltage using the given formula
        if t == '00:00':
            voltages.append(V0)  # Initial voltage
        else:
            P = model.P[t]()
            s = model.s[t]()
            V = V0 - P / ((V0)/(rho * s + 1e-9) + (V0)/(rho * (S - s + 1e-9)))
            voltages.append(V)
        
        distances.append(model.s[t]()/1000)  # Convert to km
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot voltage vs time
    ax1.plot(times, voltages, 'b-', linewidth=2)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title(f'Train Voltage Profile vs Time (S={S/1000}km)')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Plot voltage vs distance
    ax2.plot(distances, voltages, 'r-', linewidth=2)
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Voltage (V)')
    ax2.set_title(f'Train Voltage Profile vs Distance (S={S/1000}km)')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()

# Use the function to find minimum time
min_possible_time = find_minimum_time(S, delta_t, max_acc, max_p, max_v, m, C_d, A, C, theta, eta, braking_eff, min_time, max_time, limited_power_1, limited_time_start_1, limited_time_end_1, limited_power_2, limited_time_start_2, limited_time_end_2)
print(f"\nMinimum possible time: {min_possible_time} seconds")

# Also modify the final data creation after finding minimum time
total_time = round(min_possible_time * Margin/100)
speed_limits = [max_v] * total_time
power_limits_1 = create_power_limits(total_time, max_p, limited_power_1, limited_time_start_1, limited_time_end_1)  # Using same power limitation
power_limits_2 = create_power_limits(total_time, max_p, limited_power_2, limited_time_start_2, limited_time_end_2)  # Using same power limitation
data = {'00:00': {'v_max': speed_limits[0], 't_prev': '', 'P_sub1_max': power_limits_1[0], 'P_sub2_max': power_limits_2[0]},}
for i in range(1, total_time):
    minutes = i // 60
    seconds = i % 60
    time = f'{minutes:02d}:{seconds:02d}'
    prev_minutes = (i-1) // 60 
    prev_seconds = (i-1) % 60
    prev_time = f'{prev_minutes:02d}:{prev_seconds:02d}'
    data[time] = {'v_max': speed_limits[i], 't_prev': prev_time, 'P_sub1_max': power_limits_1[i], 'P_sub2_max': power_limits_2[i]}

model, termination_condition = train(rho, S, delta_t, max_acc, max_p, data, m, C_d, A, C, theta, eta, braking_eff)

if termination_condition == pyomo.TerminationCondition.optimal or termination_condition == pyomo.TerminationCondition.locallyOptimal:
    # save to file
    base_filename = f"t{total_time}_S{S}"
    ext = ".txt"
    suffix_num = 0
    filename = f"{base_filename}_e{suffix_num}{ext}"
    while os.path.exists(filename):
        suffix_num += 1
        filename = f"{base_filename}_e{suffix_num}{ext}"

    with open(filename, 'w') as f:
        f.write(f"P_sub1_max: {0} W, P_sub2_max: {0} W\n")
        f.write(f"Braking Efficiency: {braking_eff*100}%\n")
        f.write("Time(s), Velocity(km/h), Power (MW), P_sub1 (MW), P_sub2 (MW)\n")  # Header
        for t in data.keys():
            v_out = 3.6 * model.v[t]()
            if model.P[t]() < 0:
                P_out = braking_eff * model.P[t]() / 1000000 # Added braking_eff because when braking, all of the power is not going back to the grid
            else:
                P_out = model.P[t]() / 1000000
            P_sub1_out = model.P_sub1[t]() / 1000000
            P_sub2_out = model.P_sub2[t]() / 1000000
            f.write(f"{t}, {v_out:.3f}, {P_out:.6f}, {P_sub1_out:.6f}, {P_sub2_out:.6f}\n")

    plot_Pm_and_Pn_profile(model, data)
    plot_substation_powers(model, data)
    plot_voltage_profile(model, data)
    plt.show()
else:
    print(f"Infeasible - {termination_condition}")