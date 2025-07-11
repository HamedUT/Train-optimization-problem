import numpy as np
import operator
from profilesteering.opt import optAlg

class TrainBattery():
    def __init__(self, train_model, train_params, electrical_params):
        # Store reference to the train model and parameters
        self.train_model = train_model
        self.train_params = train_params
        self.electrical_params = electrical_params
        
        # ProfileSteering required variables
        self.profile = []    # Current power profile (x_m in PS paper)
        self.candidate = []  # Candidate power profile (^x_m in PS paper)
        
        # Battery parameters
        self.capacity = electrical_params.batt_cap  # Wh
        self.max_power = electrical_params.batt_max_discharge  # W
        self.min_power = -electrical_params.batt_max_charge  # W (negative for charging)
        self.initialSoC = electrical_params.batt_initial_soc * electrical_params.batt_cap
        
        # For optimization
        self.opt = optAlg.OptAlg()
        
        # Store simulation data
        self.data = None
    
    def set_data(self, data):
        """Set the simulation data (distances, grades, speed limits)"""
        self.data = data
    
    def extract_battery_profile(self):
        """Extract battery power profile from the train model"""
        if self.train_model is None:
            return []
            
        # Extract time-based power profile from distance-based model
        times = []
        powers = []
        d_keys = list(self.data.keys())
        
        for idx in range(1, len(d_keys)):
            d = d_keys[idx]
            t = self.train_model.t[d]()
            pb = self.train_model.Pb[d]()
            
            times.append(t)
            powers.append(pb)
            
        return times, powers
        
    def init(self, p):
        """Initialize the device with a desired profile"""
        # Create an empty profile of the right length
        self.profile = [0] * len(p)
        
        # Extract current battery profile if available
        if self.train_model is not None:
            times, powers = self.extract_battery_profile()
            
            # Resample to match the desired profile length if necessary
            if len(times) > 0:
                total_time = times[-1]
                time_step = total_time / (len(p) - 1)
                
                for i in range(len(p)):
                    t = i * time_step
                    # Find nearest power value
                    idx = min(range(len(times)), key=lambda j: abs(times[j] - t))
                    if idx < len(powers):
                        self.profile[i] = powers[idx]
        
        return list(self.profile)
        
    def plan(self, d):
        """Plan a new profile based on the desired profile d"""
        # Calculate difference profile
        p_m = list(map(operator.sub, self.profile, d))  # p_m = x_m - d
        
        # Use the optimization algorithm to create a new plan
        self.candidate = self.opt.bufferPlanning(
            p_m,
            self.initialSoC,  # Target SoC
            self.initialSoC,  # Initial SoC
            self.capacity,
            [0] * len(p_m),  # No static losses
            [],  # No specific charging powers
            self.min_power, 
            self.max_power,
            [], [],  # No power limits
            False,  # No reactive power
            [],  # No prices
            1  # Profile weight
        )
        
        # Calculate improvement
        e_m = np.linalg.norm(np.array(self.profile)-np.array(p_m)) - np.linalg.norm(np.array(self.candidate)-np.array(p_m))
        
        return e_m
        
    def accept(self):
        """Accept the candidate profile"""
        diff = list(map(operator.sub, self.candidate, self.profile))
        self.profile = list(self.candidate)
        return diff