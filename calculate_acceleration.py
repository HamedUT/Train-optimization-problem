def calculate_acceleration(v_kmh, max_p, eta, m, C_d, A, C, g, rho_air, WindSpeed):
    # Convert velocity from km/h to m/s
    v = v_kmh / 3.6

    # Calculate drag force
    drag_force = 0.5 * rho_air * C_d * A * (v + WindSpeed)**2

    # Calculate rolling resistance
    rolling_resistance = C * m * g

    # Calculate acceleration
    acceleration = (max_p * eta / (v + 1e-6) + drag_force + rolling_resistance) / m
    return acceleration

# Parameters
v_kmh = 90.844  # Velocity in km/h
max_p = 1393000  # Maximum power in watts
eta = 0.857  # Efficiency
m = 152743 * (1 + 0.0674)  # Train mass in kg
C_d = 0.8  # Drag coefficient
A = 2.88 * 4.25  # Frontal area in m²
C = 0.002  # Rolling resistance coefficient
g = 9.807  # Gravitational acceleration in m/s²
rho_air = 1.225  # Air density in kg/m³
WindSpeed = 0  # Wind speed in m/s

# Calculate acceleration
acceleration = calculate_acceleration(v_kmh, max_p, eta, m, C_d, A, C, g, rho_air, WindSpeed)
print(f"Acceleration at {1e-6} km/h: {acceleration:.3f} m/s²")