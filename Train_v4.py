# Added regenrative braking option in the objective function. Also calculated P of substations after model is solved.
import pyomo.environ as pyomo
import matplotlib.pyplot as plt
import numpy as np
import sys

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
eta = 0.893564 # Efficiency of the train's propulsion system
C_d = 0.8 # (Drag coefficient)
braking_eff = 0.0 # Regenerative braking efficiency
max_v = 44.444 # m/s = 160 km/h (VIRM)
max_acc = 0.768 # m/s2 (2.76 km/h/s) (VIRM)

# Enviromental Variables
total_time = 430 # (sec) From Substation 1 to Substation 2
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

def train(rho_1, rho_2, S, V0, delta_t, max_acc, max_p, data, m, C_d, A, C, theta, eta, braking_eff):

    v_opt, P_opt = Initializer(rho_1, rho_2, S, V0, delta_t, max_acc, max_p, data, m, C_d, A, C, theta, eta, braking_eff)    
    T = data.keys()
    
    model = pyomo.ConcreteModel()

    # Decision Variables
    model.v = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0,max_v), initialize=lambda model0, t: v_opt[t]) # velocity (m/s)

    # State Variables
    model.Pm = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0, max_p)) # Power consumption (kW)
    model.Pn = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0, min_p)) # Power consumption (kW)
    model.P = pyomo.Var(T, domain=pyomo.Reals, initialize=lambda model0, t: P_opt[t]) # Power consumption (kW)
    model.s = pyomo.Var(T, domain=pyomo.NonNegativeReals) # Distance
    # model.d = pyomo.Var(T, domain=pyomo.Binary)
    #model.V = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0, V0-1)) # Voltage
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

    for t in list(T)[1:]:
        model.cons.add(model.v[t] == (model.s[t] - model.s[data[t]['t_prev']]) / delta_t)
        model.cons.add((model.v[t] - model.v[data[t]['t_prev']])/delta_t <= max_acc)
        model.cons.add(-max_acc <= (model.v[t] - model.v[data[t]['t_prev']])/delta_t)
        model.cons.add(model.P[t] == model.Pm[t] - model.Pn[t])
        #model.cons.add(model.R1[t] == (rho_1) * model.s[t] + 0.000000000001)
        #model.cons.add(model.R2[t] == (rho_2) * (S - model.s[t]) + 0.000000000001)
        #model.cons.add(model.P[t] == model.V[t] * ((V0 - model.V[t])/(model.R1[t]) + (V0 - model.V[t])/(model.R2[t])))

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
    results = solver.solve(model, tee=False)


    # Display resultsf"{number:.3f}"
    for t in data.keys():
        v_out = 3.6*model.v[t]()
        s_out = model.s[t]()/1000
        p_out = model.P[t]()/1000000
        #V_out = model.V[t]()

        if t == '00:00':
            #print('  ', t, ':', f"{v_out:.3f}", 'km/h','  ', model.s[t]()/1000, 'km','  ', 3.6*(model.v[t]()), 'km/h/s','  ', (model.P[t]()), 'W','  ', (model.V[t]()), 'Volts')
            print('  ', t, ':', f"{v_out:.3f}", 'km/h','  ', model.s[t]()/1000, 'km','  ', 3.6*(model.v[t]()), 'km/h/s','  ', (model.P[t]()), 'W')
        else:
            a_out = (model.v[t]() - model.v[data[t]['t_prev']]())
            #print('  ', t, ':', f"{v_out:.2f}", 'km/h','  ', f"{s_out:.3f}", 'km','  ', f"{a_out:.2f}", 'km/h/s','  ', f"{p_out:.0f}", 'W','  ', f"{V_out:.1f}", 'Volts')
            print('  ', t, ':', f"{v_out:.2f}", 'km/h','  ', f"{s_out:.3f}", 'km','  ', f"{a_out:.4f}", 'm/s2','  ', f"{p_out:.6f}", 'MW')
    print('Value of O.F. = {:.3f} kWh'.format((model.of())/3600000*total_time))
    
    # save to file
    # filename = f'velocity_output_t{total_time}_S{S}.txt'
    # with open(filename, 'w') as f:
    #     f.write("Time(s), Velocity(km/h)\n")  # Header
    #     for t in data.keys():
    #         v_out = 3.6*model.v[t]()
    #         f.write(f"{t}, {v_out:.3f}\n")
    
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
    
def plot_substation_powers(model, data, rho_1, rho_2, S):
    """Calculate and plot power from each substation based on train position and total power"""
    times = []
    P_sub1_values = []
    P_sub2_values = []
    distances = []
    
    for t in data.keys():
        minutes, seconds = map(int, t.split(':'))
        times.append(minutes*60 + seconds)
        
        # Calculate resistances based on position
        R1 = rho_1 * model.s[t]()
        R2 = rho_2 * (S - model.s[t]())
        
        # Calculate power split using voltage divider principle
        P_total = model.P[t]()
        if P_total > 0:  # Only split positive power
            P_sub1 = P_total * (R2/(R1 + R2))
            P_sub2 = P_total * (R1/(R1 + R2))
        else:
            P_sub1 = 0
            P_sub2 = 0
            
        P_sub1_values.append(P_sub1/1000000)  # Convert to MW
        P_sub2_values.append(P_sub2/1000000)  # Convert to MW
        distances.append(model.s[t]()/1000)  # Convert to km
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot power vs time
    ax1.plot(times, P_sub1_values, 'b-', label='Substation 1')
    ax1.plot(times, P_sub2_values, 'r-', label='Substation 2')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Power (MW)')
    ax1.set_title('Substation Power Profiles')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot power vs distance
    ax2.plot(distances, P_sub1_values, 'b-', label='Substation 1')
    ax2.plot(distances, P_sub2_values, 'r-', label='Substation 2')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Power (MW)')
    ax2.set_title('Substation Power vs Distance')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    
    # Calculate total energy from each substation
    E_sub1 = sum(p * delta_t for p in P_sub1_values) / 3600  # MWh
    E_sub2 = sum(p * delta_t for p in P_sub2_values) / 3600  # MWh
    
    print(f"\nEnergy consumption summary:")
    print(f"Substation 1: {E_sub1:.2f} MWh")
    print(f"Substation 2: {E_sub2:.2f} MWh")
    print(f"Total: {(E_sub1 + E_sub2):.2f} MWh")

model, termination_condition = train(rho_1, rho_2, S, V0, delta_t, max_acc, max_p, data, m, C_d, A, C, theta, eta, braking_eff)
if termination_condition == pyomo.TerminationCondition.optimal or termination_condition == pyomo.TerminationCondition.locallyOptimal:
    # print(f"Total Time: {total_time}, Status: Feasible (Optimal)")
    # Remove plt.show() calls from within these functions.
    # Then, call both functions and finally call plt.show() to display all figures.
    plot_Pm_and_Pn_profile(model, data)
    plot_substation_powers(model, data, rho_1, rho_2, S)
    plt.show()
else:
    print(f"Infeasible - {termination_condition}")


