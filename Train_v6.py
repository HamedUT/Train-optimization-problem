# v3: Added Pm and Pn so the model decides for itself how much power to put
# v4: Added regenrative braking option in the objective function. Also calculated P of substations after model is solved.
# v5: Added P of substations as variables in the model. Also added braking_eff when writing to the txt file for further plot comparison. because when braking, all of the power is not going back to the grid
# v6: Added Voltage (After the solver and not in the model). On the Energy version, energy is the decision variable instead of velocity. no difference on the results.

import pyomo.environ as pyomo
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Electrical Parameters
rho_1 = 0.00003 # Ohms/m
rho_2 = 0.00003 # Ohms/m
V0 = 1500 # V

# Train Parameters
m = 390000 # kg (train weight, but should be variable later)
A = 3.020*4.670 # m^2 (Frontal area)
C = 0.002 # (Rolling resistance coefficient)
max_p = 2157000 # W (max power)
min_p = max_p * 2 # W (min power)
max_p_sub1 = max_p * 1.0
max_p_sub2 = max_p * 1
eta = 0.893564 # Efficiency of the train's propulsion system
C_d = 0.8 # (Drag coefficient)
braking_eff = 0.0 # Regenerative braking efficiency    
max_v = 44.444 # m/s = 160 km/h (VIRM)
max_acc = 0.768 # m/s2 (2.76 km/h/s) (VIRM)

# Enviromental Variables
total_time = 600 # (sec) From Substation 1 to Substation 2
S = 10000 # (m) Length between Substation 1 and Substation 2
delta_t = 1 # Seconds
theta = 0.004 # (Gradient)
WindSpeed = 0 # m/s (Wind speed)

data = {'00:00': {'v_max': max_v, 't_prev': ''},}

for i in range(1, total_time):
    minutes = i // 60
    seconds = i % 60
    time = f'{minutes:02d}:{seconds:02d}'
    prev_minutes = (i-1) // 60
    prev_seconds = (i-1) % 60
    prev_time = f'{prev_minutes:02d}:{prev_seconds:02d}'
    data[time] = {'v_max': max_v, 't_prev': prev_time}

def Initializer(rho_1, rho_2, S, V0, delta_t, max_acc, max_p, data, m, C_d, A, C, theta, eta, braking_eff):
    T = data.keys()
    model0 = pyomo.ConcreteModel()
    model0.v = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0,max_v)) # velocity (m/s)
    model0.P = pyomo.Var(T, domain=pyomo.Reals, bounds=(-min_p, max_p)) # Power consumption (kW)
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
        print(f"Infeasible during Initialization - {termination_condition}")
        sys.exit()
    return v_opt, P_opt

def train(rho_1, rho_2, S, V0, delta_t, max_acc, max_p, data, m, C_d, A, C, theta, eta, braking_eff, max_p_sub1, max_p_sub2):

    v_opt, P_opt = Initializer(rho_1, rho_2, S, V0, delta_t, max_acc, max_p, data, m, C_d, A, C, theta, eta, braking_eff)    
    T = data.keys()
    
    model = pyomo.ConcreteModel()

    # Decision Variables
    model.v = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0,max_v), initialize=lambda model0, t: v_opt[t]) # velocity (m/s)

    # State Variables
    model.Pm = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0, max_p)) 
    model.Pn = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0, min_p)) 
    model.P = pyomo.Var(T, domain=pyomo.Reals, initialize=lambda model0, t: P_opt[t])
    model.P_sub1 = pyomo.Var(T, domain=pyomo.Reals, bounds=(-max_p_sub1, max_p_sub1)) 
    model.P_sub2 = pyomo.Var(T, domain=pyomo.Reals, bounds=(-max_p_sub2, max_p_sub2))
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
        model.cons.add(model.P_sub1[t] == model.P[t] * ((1e-09 + rho_2 * (S - model.s[t]))/(1e-09 + rho_1 * model.s[t] + rho_2 * (S - model.s[t]))))
        model.cons.add(model.P_sub2[t] == model.P[t] * ((1e-09 + rho_1 * model.s[t])/(1e-09 + rho_1 * model.s[t] + rho_2 * (S - model.s[t]))))
        #model.cons.add(model.R1[t] == (rho_1) * model.s[t] + 0.000000000001)
        #model.cons.add(model.R2[t] == (rho_2) * (S - model.s[t]) + 0.000000000001)

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

            
    # Define solver
    solver = pyomo.SolverFactory('ipopt')
    
    # Set solver options
    # solver.options['max_iter'] = 15000
    # solver.options['tol'] = 1e-6
    # solver.options['acceptable_tol'] = 1e-5
    # solver.options['mu_init'] = 1e-1

    # Solve the model
    results = solver.solve(model, tee=True)


    # Display resultsf"{number:.3f}"
    for t in data.keys():
        v_out = 3.6*model.v[t]()
        s_out = model.s[t]()/1000
        p_out = model.P[t]()/1000000
        # V_out = model.V[t]()

        if t == '00:00':
            #print('  ', t, ':', f"{v_out:.3f}", 'km/h','  ', model.s[t]()/1000, 'km','  ', 3.6*(model.v[t]()), 'km/h/s','  ', (model.P[t]()), 'W','  ', (model.V[t]()), 'Volts')
            print('  ', t, ':', f"{v_out:.3f}", 'km/h','  ', model.s[t]()/1000, 'km','  ', 3.6*(model.v[t]()), 'km/h/s','  ', (model.P[t]()), 'W')
        else:
            a_out = (model.v[t]() - model.v[data[t]['t_prev']]())
            #print('  ', t, ':', f"{v_out:.2f}", 'km/h','  ', f"{s_out:.3f}", 'km','  ', f"{a_out:.2f}", 'km/h/s','  ', f"{p_out:.0f}", 'W','  ', f"{V_out:.1f}", 'Volts')
            print('  ', t, ':', f"{v_out:.2f}", 'km/h','  ', f"{s_out:.3f}", 'km','  ', f"{a_out:.4f}", 'm/s2','  ', f"{p_out:.6f}", 'MW')
    print('Value of O.F. = {:.3f} kWh'.format((model.of())/3600000*total_time))
     
    return model, results.solver.termination_condition  # Add return statement  # Add return statement

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
            V = V0 - P / ((V0)/(rho_1 * s + 1e-9) + (V0)/(rho_2 * (S - s + 1e-9)))
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

model, termination_condition = train(rho_1, rho_2, S, V0, delta_t, max_acc, max_p, data, m, C_d, A, C, theta, eta, braking_eff, max_p_sub1, max_p_sub2)

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
        
        f.write(f"P_sub1_max: {max_p_sub1} W, P_sub2_max: {max_p_sub2} W\n")
        f.write(f"Braking Efficiency: {braking_eff*100}%\n")
        f.write("Time(s), Velocity(km/h), Power (MW)\n")  # Header
        for t in data.keys():
            v_out = 3.6 * model.v[t]()
            if model.P[t]() < 0:
                P_out = braking_eff * model.P[t]() / 1000000 # Added braking_eff because when braking, all of the power is not going back to the grid
            else:
                P_out = model.P[t]() / 1000000
            f.write(f"{t}, {v_out:.3f}, {P_out:.6f}\n")

    plot_Pm_and_Pn_profile(model, data)
    plot_substation_powers(model, data)
    plot_voltage_profile(model, data)
    plt.show()
else:
    print(f"Infeasible - {termination_condition}")