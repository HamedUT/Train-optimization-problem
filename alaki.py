import pyomo.environ as pyomo
import matplotlib.pyplot as plt
import numpy as np
import random
import time

# ---------------------------
# Global Parameters
# ---------------------------
rho = 0.00003  # Ohm/m
V0 = 1500  # V
m = 150000  # kg
A = 2.88 * 4.30  # m²
C = 0.002  # Rolling resistance coefficient
eta = 0.893564  # Efficiency
C_d = 0.8  # Drag coefficient
braking_eff = 0.1
max_v = 44.444  # m/s
max_acc = 0.768  # m/s²
max_braking = 0.4  # m/s²
max_p = 2157000  # W
delta_s = 250  # m
total_distance = 38500  # m
total_time = 24 * 60  # sec
WindSpeed = 2.5  # m/s

v_init = 0.0  # Initial velocity (m/s)
t_init = 0.0  # Initial time (s)

max_gradient = 0.0015  # Constant gradient
gradients = [max_gradient] * (int(total_distance / delta_s) + 1)

# ---------------------------
# Helper Functions
# ---------------------------
def resistive_force(v, grade, wind):
    """ Calculate total resistive force """
    Fr = m * 9.81 * (C + grade) + 0.5 * 1.225 * A * C_d * (v - wind)**2
    return Fr

def Initializer(D, delta_s, v0, max_v, v_init_guess):
    model = pyomo.ConcreteModel()
    
    model.v = pyomo.Var(D, domain=pyomo.NonNegativeReals, bounds=(0, max_v))  # velocity
    model.P = pyomo.Var(D, domain=pyomo.Reals)  # power
    model.cons = pyomo.ConstraintList()
    model.obj = pyomo.Objective(expr=0, sense=pyomo.minimize)

    first_d = list(D)[0]
    last_d = list(D)[-1]
    
    model.v[first_d].fix(v0)  # Fix initial velocity
    model.P[first_d].fix(0)   # Assume initial power is 0

    for i in range(1, len(D)):
        d_now = D[i]
        d_prev = D[i-1]
        
        v_now = model.v[d_now]
        v_prev = model.v[d_prev]
        
        # Discrete acceleration constraint
        model.cons.add((v_now - v_prev) / (2 * delta_s / (v_now + v_prev)) <= max_acc)
        model.cons.add(-(v_now - v_prev) / (2 * delta_s / (v_now + v_prev)) <= max_braking)

    # Set up the objective (energy minimization)
    energy_expr = 0
    for i in range(1, len(D)):
        d_now = D[i]
        d_prev = D[i-1]
        
        avg_v = (model.v[d_now] + model.v[d_prev]) / 2
        delta_t = 2 * delta_s / (model.v[d_now] + model.v[d_prev] + 1e-6)  # avoid division by zero
        acc = (model.v[d_now]**2 - model.v[d_prev]**2) / (2 * delta_s)
        F_total = resistive_force(avg_v, gradients[min(d_now // delta_s, len(gradients)-1)], WindSpeed) + m * acc
        
        if F_total >= 0:
            P_mech = F_total * avg_v
            P_elec = P_mech / eta
        else:
            P_mech = F_total * avg_v
            P_elec = braking_eff * P_mech / eta
        
        energy_expr += P_elec * delta_t

    model.obj.set_value(energy_expr)

    return model

# ---------------------------
# Rolling Horizon Setup
# ---------------------------
window_size = 6  # number of steps inside each window (can tune)
overlap = 1  # how many steps overlap

current_distance = 0
current_velocity = v_init
current_time = t_init

distance_list = [0]
velocity_list = [v_init]
time_list = [t_init]
power_list = [0]

while current_distance < total_distance:
    remaining_distance = total_distance - current_distance
    steps_ahead = min(window_size, int(remaining_distance / delta_s))

    D = [current_distance + i * delta_s for i in range(steps_ahead + 1)]
    D = [round(d) for d in D]  # avoid floating points error

    print(f"Rolling Horizon Window starting at distance {current_distance} m with velocity {current_velocity:.2f} m/s")

    model = Initializer(D, delta_s, current_velocity, max_v, current_velocity)
    solver = pyomo.SolverFactory('ipopt')
    solver.options['tol'] = 1e-4
    solver.options['max_iter'] = 500
    result = solver.solve(model, tee=False)

    if (result.solver.status != pyomo.SolverStatus.ok) or (result.solver.termination_condition != pyomo.TerminationCondition.optimal):
        print("Solver failed at distance", current_distance)
        break

    # Extract results
    v_next = []
    p_next = []
    for d in D:
        v_next.append(pyomo.value(model.v[d]))
        p_next.append(pyomo.value(model.P[d]))

    # Append results (but only non-overlapping parts)
    for i in range(1, len(D) - overlap):
        delta_t = 2 * delta_s / (v_next[i] + v_next[i-1] + 1e-6)  # small epsilon
        avg_v = (v_next[i] + v_next[i-1]) / 2
        distance_list.append(D[i])
        velocity_list.append(v_next[i])
        time_list.append(time_list[-1] + delta_t)
        power_list.append(p_next[i])

    # Update initial condition for next window
    current_distance = distance_list[-1]
    current_velocity = velocity_list[-1]
    current_time = time_list[-1]

# ---------------------------
# Plotting
# ---------------------------
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(distance_list, velocity_list, label='Velocity Profile')
plt.xlabel('Distance (m)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(distance_list, power_list, label='Power Profile')
plt.xlabel('Distance (m)')
plt.ylabel('Power (W)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Total Trip Time: {time_list[-1]/60:.2f} min")
