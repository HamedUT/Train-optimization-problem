import pyomo.environ as pyomo

data = {
    '00:00': {'v_avg' : 100, 't_prev' : ''}, # km/h
    '00:01': {'v_avg' : 100, 't_prev' : '00:00'},
    '00:02': {'v_avg' : 100, 't_prev' : '00:01'},
    '00:03': {'v_avg' : 100, 't_prev' : '00:02'},
    '00:04': {'v_avg' : 100, 't_prev' : '00:03'},
    '00:05': {'v_avg' : 120, 't_prev' : '00:04'},
    '00:06': {'v_avg' : 120, 't_prev' : '00:05'},
    '00:07': {'v_avg' : 120, 't_prev' : '00:06'},
    '00:08': {'v_avg' : 120, 't_prev' : '00:07'},
    '00:09': {'v_avg' : 120, 't_prev' : '00:08'},
}

rho_1 = 0.03 # Ohms/km
rho_2 = 0.05 # Ohms/km
S = 10 # km
V0 = 1500 # V
delta_t = 1/60 # hours
bs_constant = 0.01 # km/h / kW
max_acc = 3 # km/h

def train_example(rho_1, rho_2, S, V0, delta_t, bs_constant, max_acc,data):
    # Sets
    T = data.keys()

    # Type of model
    model = pyomo.ConcreteModel()

    # Decision Variables
    model.v = pyomo.Var(T, domain=pyomo.NonNegativeReals, initialize=100) # velocity (km/h)
    # State Variables
    model.P = pyomo.Var(T, domain=pyomo.Reals) # Power consumption (kW)
    model.V = pyomo.Var(T, domain=pyomo.NonNegativeReals) # Voltage
    model.s = pyomo.Var(T, domain=pyomo.NonNegativeReals) # Distance
    model.R1 = pyomo.Var(T, domain=pyomo.NonNegativeReals) # Resistance branch 1
    model.R2 = pyomo.Var(T, domain=pyomo.NonNegativeReals) # Resistance branch 2

    # Objective function
    model.of = pyomo.Objective(expr = sum((model.v[t] - data[t]['v_avg'])**2 for t in T))

    # Constraints
    model.cons = pyomo.ConstraintList()
    for t in T:
        model.cons.add(model.P[t]*1000 == model.V[t] * ((V0 - model.V[t])/(model.R1[t]) + (V0 - model.V[t])/(model.R2[t])))
        model.cons.add(model.R1[t] == (rho_1) * model.s[t])
        model.cons.add(model.R2[t] == (rho_2) * (S - model.s[t]))
        model.cons.add(model.P[t] == bs_constant * model.v[t]) # Nonsense
        model.cons.add(model.v[t] == model.s[t] / delta_t)
    # Initial conditions
    model.cons.add(model.v['00:00'] == 1) # cant be 0, because we set it to nonnegativereals
    
    for t in T:
        model.cons.add(model.s[t] <= S)
        if t != '00:00':
            model.cons.add(model.v[t] - model.v[data[t]['t_prev']] <= max_acc)
            model.cons.add(-max_acc <= model.v[t] - model.v[data[t]['t_prev']])
    

    # Define solver
    solver = pyomo.SolverFactory('ipopt')
    solver.solve(model, tee=True)

    # Display results
    for t in data.keys():
        print('  ', t, ':', model.v[t](), 'km/h')
    print('Value of O.F. = ', model.of())

train_example(rho_1, rho_2, S, V0, delta_t, bs_constant, max_acc,data)