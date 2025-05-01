class TrainParameters:
    def __init__(self):
        # Electrical Parameters
        self.rho = 0.00003  # Ohms/m
        self.V0 = 1500      # V

        # Train Parameters
        self.mass = 390000          # kg (train weight)
        self.frontal_area = 3.020 * 4.670  # m^2
        self.rolling_resistance = 0.002     # coefficient
        self.max_power = 2157000    # W
        self.min_power = self.max_power * 2
        self.max_p_sub1 = self.max_power * 1.0
        self.max_p_sub2 = self.max_power * 5
        self.efficiency = 0.893564
        self.drag_coef = 0.8
        self.braking_eff = 0.0
        self.max_velocity = 44.444  # m/s = 160 km/h
        self.max_acceleration = 0.768  # m/s2

        # Environmental Variables
        self.total_time = 500    # sec
        self.distance = 10000    # m
        self.time_step = 1       # sec
        self.gradient = 0.004    # slope
        self.wind_speed = 0      # m/s

        # Battery Parameters
        self.max_energy_kwh = 278
        self.max_energy = self.max_energy_kwh * 3600000  # Convert to Joules
        self.init_soc = 0.5 * self.max_energy
        self.min_soc = 0.2 * self.max_energy
        self.max_soc = self.max_energy
        self.max_battery_power = 500000  # W

# Helper function to load speed and power limits
def load_limits():
    speed_limits = []
    power_limits = []
    
    with open('speed_limits.txt', 'r') as f:
        speed_limits = [float(line.strip()) for line in f]
        
    with open('power_limits.txt', 'r') as f:
        power_limits = [float(line.strip()) for line in f]
        
    return speed_limits, power_limits

# Function to create time-based data dictionary
def create_time_data(total_time, speed_limits, power_limits):
    data = {'00:00': {'v_max': speed_limits[0], 't_prev': '', 'P_sub1_max': power_limits[0]}}
    
    for i in range(1, total_time):
        minutes = i // 60
        seconds = i % 60
        time = f'{minutes:02d}:{seconds:02d}'
        prev_minutes = (i-1) // 60 
        prev_seconds = (i-1) % 60
        prev_time = f'{prev_minutes:02d}:{prev_seconds:02d}'
        data[time] = {
            'v_max': speed_limits[i], 
            't_prev': prev_time, 
            'P_sub1_max': power_limits[i]
        }
    
    return data