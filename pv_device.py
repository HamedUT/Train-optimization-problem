import numpy as np
import operator
import math
from datetime import datetime
from profilesteering.opt import optAlg

class PVDevice:
    def __init__(self, peak_power=1e6, location="Netherlands"):
        # Store PV system parameters
        self.peak_power = peak_power  # Peak power in W (1 MW default)
        self.location = location       # Location affects solar irradiance pattern
        
        # ProfileSteering required variables
        self.profile = []    # Current power profile (x_m in PS paper)
        self.candidate = []  # Candidate power profile (^x_m in PS paper)
        
        # For optimization
        self.opt = optAlg.OptAlg()
        
        # PV generation parameters
        self.sunrise_hour = 6.0   # 6:00 AM
        self.sunset_hour = 20.0   # 8:00 PM
        self.peak_hour = 13.0     # 1:00 PM (solar noon)
        self.cloud_factor = 0.8   # Random cloud cover factor (0.5 to 1.0)
        
        # Set day parameters - can be changed to model specific days
        self.day_of_year = datetime.now().timetuple().tm_yday  # Current day of year
        self.total_day_hours = self.sunset_hour - self.sunrise_hour
    
    def generate_pv_profile(self, num_intervals, total_time):
        """Generate a realistic PV production profile based on time of day"""
        # Time step in hours
        time_step_hours = total_time / 3600 / (num_intervals - 1)
        
        # Starting time (assume simulation starts at specific time)
        start_hour = 8.0  # 8:00 AM default - can be parameterized
        
        # Calculate PV production for each interval
        pv_profile = []
        for i in range(num_intervals):
            current_hour = start_hour + i * time_step_hours
            
            # No production before sunrise or after sunset
            if current_hour < self.sunrise_hour or current_hour > self.sunset_hour:
                pv_profile.append(0)
                continue
            
            # Calculate relative position in daylight hours (0 at sunrise, 1 at sunset)
            day_position = (current_hour - self.sunrise_hour) / self.total_day_hours
            
            # Bell curve with peak at solar noon
            # Using modified cosine function for smoother curve
            sun_intensity = math.cos((day_position - 0.5) * math.pi) ** 2
            
            # Apply random cloud variations
            cloud_factor = self.cloud_factor * (0.8 + 0.2 * np.random.random())
            
            # Calculate actual power output (NEGATIVE for generation)
            power_output = -self.peak_power * sun_intensity * cloud_factor
            
            pv_profile.append(power_output)
        
        return pv_profile
        
    def init(self, p):
        """Initialize the device with a desired profile"""
        # Generate a PV profile for the given number of intervals
        # We'll ignore the desired profile since PV is not controllable
        self.profile = self.generate_pv_profile(len(p), 8 * 3600)  # Assuming 8 hour simulation
        
        # Return the initial PV generation profile
        return list(self.profile)
    
    def plan(self, d):
        """Plan a new profile based on the desired profile d"""
        # Calculate difference profile
        p_m = list(map(operator.sub, self.profile, d))
        
        # PV output is not controllable, so candidate profile stays the same
        # We could add some minor adjustments for curtailment if necessary
        self.candidate = list(self.profile)
        
        # Calculate improvement (should be 0 for non-controllable PV)
        e_m = np.linalg.norm(np.array(self.profile)-np.array(p_m)) - np.linalg.norm(np.array(self.candidate)-np.array(p_m))
        
        return e_m
        
    def accept(self):
        """Accept the candidate profile"""
        diff = list(map(operator.sub, self.candidate, self.profile))
        self.profile = list(self.candidate)
        return diff