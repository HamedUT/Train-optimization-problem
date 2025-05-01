import pyomo.environ as pyomo
import matplotlib.pyplot as plt
import numpy as np

# Electrical Parameters
rho_1 = 0.00003 # Ohms/m
rho_2 = 0.00003 # Ohms/m
V0 = 1500 # V

# Train Parameters
m = 390000 # kg (train weight, but should be variable later)
A = 3.020*4.670 # m^2 (Frontal area)
C = 0.002 # (Rolling resistance coefficient)
max_p = 2157000 # W (max power)
eta = 0.893564 # Efficiency of the train's propulsion system
C_d = 0.8 # (Drag coefficient)
braking_eff = 0.1 # Regenerative braking efficiency
max_v = 44.444 # m/s = 160 km/h (VIRM)
max_acc = 0.768 # m/s2 (2.76 km/h/s) (VIRM)

# Enviromental Variables
total_time = 228 # (sec) From Substation 1 to Substation 2
S = 5000 # (m) Length between Substation 1 and Substation 2
delta_t = 1 # Seconds
theta = 0.004 # (Gradient)
WindSpeed = 0 # m/s (Wind speed)

data = {
    '00:00': {'v_max': max_v, 't_prev': ''},
}

for i in range(1, total_time):
    minutes = i // 60
    seconds = i % 60
    time = f'{minutes:02d}:{seconds:02d}'
    prev_minutes = (i-1) // 60
    prev_seconds = (i-1) % 60
    prev_time = f'{prev_minutes:02d}:{prev_seconds:02d}'
    data[time] = {'v_max': max_v, 't_prev': prev_time}

def train(rho_1, rho_2, S, V0, delta_t, max_acc, max_p, data, m, C_d, A, C, theta, eta, braking_eff):
    # Sets
    T = data.keys()

    # Type of model
    model = pyomo.ConcreteModel()

    # Decision Variables
    model.v = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0,max_v)) # velocity (m/s)

    # State Variables
    model.P = pyomo.Var(T, domain=pyomo.Reals, bounds=(-10*max_p, max_p)) # Power consumption (kW)
    model.s = pyomo.Var(T, domain=pyomo.NonNegativeReals) # Distance
    #model.V = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0, V0-1)) # Voltage
    #model.R1 = pyomo.Var(T, domain=pyomo.NonNegativeReals) # Resistance branch 1
    #model.R2 = pyomo.Var(T, domain=pyomo.NonNegativeReals) # Resistance branch 2

    # Objective function Detailed version, works properly but doesn't change the results that much.
    # model.of = pyomo.Objective(expr = sum(pyomo.Expr_if(model.P[t] >= 0, model.P[t]**2, (-braking_eff*(model.P[t] - C * m * 9.807 * model.v[t] - 0.5 * 1.225 * C_d * A * model.v[t]**3))**2) for t in T))

    # Objective function Simplified version
    model.of = pyomo.Objective(expr = sum(pyomo.Expr_if(model.P[t] >= 0, model.P[t]**2, (-braking_eff*(model.P[t]))**2) for t in T))

    # Constraints
    model.cons = pyomo.ConstraintList()
    
    # Ensure the train reaches the destination
    final_time = list(T)[-1]
    model.cons.add(model.s[final_time] == S)
    
    # Initial conditions
    model.v['00:00'].fix(0)
    model.v[final_time].fix(0)
    model.s['00:00'].fix(0)
    model.P['00:00'].fix(0)


    for t in list(T)[1:]:
        model.cons.add(model.v[t] == (model.s[t] - model.s[data[t]['t_prev']]) / delta_t)
        model.cons.add((model.v[t] - model.v[data[t]['t_prev']])/delta_t <= max_acc)
        model.cons.add(-max_acc <= (model.v[t] - model.v[data[t]['t_prev']])/delta_t)
        # model.cons.add(model.v[t] <= data[t]['v_max'])
        #model.cons.add(model.s[t] <= S)
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
            0.5 * 1.225 * C_d * A * model.v[t]**2 +
            C * m * 9.807 +
            m * 9.807 * theta +
            m * (model.v[t] - model.v[data[t]['t_prev']])/delta_t
            )* model.v[t])    

            
    # Define solver
    solver = pyomo.SolverFactory('ipopt')
    
    # Set solver options
    # solver.options['max_iter'] = 15000
    solver.options['tol'] = 1e-9
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
            a_out = 3.6*(model.v[t]() - model.v[data[t]['t_prev']]())
            #print('  ', t, ':', f"{v_out:.2f}", 'km/h','  ', f"{s_out:.3f}", 'km','  ', f"{a_out:.2f}", 'km/h/s','  ', f"{p_out:.0f}", 'W','  ', f"{V_out:.1f}", 'Volts')
            print('  ', t, ':', f"{v_out:.2f}", 'km/h','  ', f"{s_out:.3f}", 'km','  ', f"{a_out:.2f}", 'km/h/s','  ', f"{p_out:.3f}", 'MW')
    print('Value of O.F. = {:.3f} kW/h'.format((model.of()**0.5)/3600000))
    
    # save to file
    filename = f'velocity_output_t{total_time}_S{S}.txt'
    with open(filename, 'w') as f:
        f.write("Time(s), Velocity(km/h)\n")  # Header
        for t in data.keys():
            v_out = 3.6*model.v[t]()
            f.write(f"{t}, {v_out:.3f}\n")
    
    return model,results.solver.termination_condition  # Add return statement

def plot_velocity_profile(model, data):
    # Extract time and velocity data
    times = []
    velocities = []
    
    for t in data.keys():
        # Convert MM:SS to seconds for x-axis
        minutes, seconds = map(int, t.split(':'))
        times.append(minutes*60 + seconds)
        # Convert velocity to km/h
        velocities.append(3.6*model.v[t]())
    
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
    plt.savefig(f'velocity_profile_S{S}.png', dpi=300, bbox_inches='tight')
    plt.show()

model, termination_condition = train(rho_1, rho_2, S, V0, delta_t, max_acc, max_p, data, m, C_d, A, C, theta, eta, braking_eff)
if termination_condition == pyomo.TerminationCondition.optimal or termination_condition == pyomo.TerminationCondition.locallyOptimal:
    print(f"Total Time: {total_time}, Status: Feasible (Optimal)")
    plot_velocity_profile(model, data)
else:
    print(f"Total Time: {total_time}, Status: Infeasible or Suboptimal - {termination_condition}")

