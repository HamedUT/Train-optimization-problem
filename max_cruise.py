import numpy as np

def compute_cruise_profile(params):
    """
    Compute bang-bang profile (max acceleration, cruise, max braking)
    """
    v_init = params['v_init']
    max_v = params['max_v']
    max_acc = params['max_acc']
    max_braking = params['max_braking']
    total_distance = params['total_distance']
    total_time = params['total_time']
    delta_s = params['delta_s']

    # Calculate distances for each phase
    t_acc = (max_v - v_init) / max_acc
    s_acc = v_init * t_acc + 0.5 * max_acc * t_acc**2
    
    t_brake = max_v / max_braking
    s_brake = 0.5 * max_v * t_brake
    
    s_cruise = total_distance - s_acc - s_brake
    
    # If s_cruise < 0, need to find lower cruise velocity
    if s_cruise < 0:
        # Solve quadratic equation for max achievable velocity
        a = 1/(2*max_acc) + 1/(2*max_braking)
        b = v_init/max_acc
        c = -total_distance
        max_v = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        
        # Recalculate phases
        t_acc = (max_v - v_init) / max_acc
        s_acc = v_init * t_acc + 0.5 * max_acc * t_acc**2
        t_brake = max_v / max_braking
        s_brake = 0.5 * max_v * t_brake
        s_cruise = total_distance - s_acc - s_brake

    # Generate distance points
    distances = np.arange(0, total_distance + delta_s, delta_s)
    velocities = np.zeros_like(distances)
    times = np.zeros_like(distances)
    
    # Fill velocity profile
    for i, d in enumerate(distances):
        if d <= s_acc:
            # Acceleration phase
            t = (-v_init + np.sqrt(v_init**2 + 2*max_acc*d))/max_acc
            velocities[i] = v_init + max_acc * t
            times[i] = t
        elif d <= s_acc + s_cruise:
            # Cruise phase
            velocities[i] = max_v
            times[i] = t_acc + (d - s_acc)/max_v
        else:
            # Braking phase
            d_in_brake = d - (s_acc + s_cruise)
            velocities[i] = np.sqrt(max(max_v**2 - 2*max_braking*d_in_brake, 0))
            if velocities[i] > 0:
                times[i] = t_acc + s_cruise/max_v + (max_v - velocities[i])/max_braking

    # Compute energy using Davies equation
    energy = 0
    power = np.zeros_like(distances)
    for i in range(1, len(distances)):
        d = distances[i]
        v = velocities[i]
        v_prev = velocities[i-1]
        d_prev = distances[i-1]
        
        v_avg = (v + v_prev) / 2
        if v_avg > 0:
            dt = (d - d_prev) / v_avg
            
            # Davies equation components
            aero_force = 0.5 * 1.225 * params['C_d'] * params['A'] * (v_avg + params['WindSpeed'])**2
            rolling_force = params['C'] * params['m'] * 9.807
            grade_force = params['m'] * 9.807 * params['gradients'][i]
            acc_force = params['m'] * (v - v_prev) / dt if dt > 0 else 0
            curve_force = params['mu_curve'] * params['m'] * v_avg**2 / params['track_radius'][i]
            
            total_force = aero_force + rolling_force + grade_force + acc_force + curve_force
            power[i] = total_force * v / params['eta']
            energy += power[i] * dt / 3.6e6  # Convert to kWh

    return {
        'distances': distances,
        'velocities': velocities,
        'times': times,
        'total_energy': energy,
        'power': power  # Add power to returned dictionary
    }
