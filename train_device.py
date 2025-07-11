import numpy as np
import operator
import random
from profilesteering.opt import optAlg

class TrainDevice():
    def __init__(self, train_model, train_params, electrical_params, simulation_params):
        # Store reference to the train model and parameters
        self.train_model = train_model
        self.train_params = train_params
        self.electrical_params = electrical_params
        self.simulation_params = simulation_params
        
        # ProfileSteering required variables
        self.profile = []    # Current power profile (x_m in PS paper)
        self.candidate = []  # Candidate power profile (^x_m in PS paper)
        
        # For optimization
        self.opt = optAlg.OptAlg()
        
        # Store simulation data
        self.data = None
        self.delta_t = 1  # Time step in seconds (will be calculated based on simulation)
    
    def set_data(self, data):
        """Set the simulation data (distances, grades, speed limits)"""
        self.data = data
        
    def extract_power_profile(self):
        """Extract power profile from the train model"""
        if self.train_model is None:
            return []
            
        # Extract time-based power profile from distance-based model
        times = []
        powers = []
        d_keys = list(self.data.keys())
        
        for idx in range(1, len(d_keys)):
            d = d_keys[idx]
            t = self.train_model.t[d]()
            pm = self.train_model.Pm[d]()
            pn = self.train_model.Pn[d]()
            pb = self.train_model.Pb[d]()
            
            # Net power (excluding battery) in W
            power = pm/self.train_params.eta - self.train_params.braking_eff * pn
            
            times.append(t)
            powers.append(power)
            
        return times, powers
        
    def init(self, p):
        """Initialize the device with a desired profile"""
        # Store the initial profile length
        self.profile = [0] * len(p)
        
        # Extract current power profile
        if self.train_model is not None:
            times, powers = self.extract_power_profile()
            
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
        
        # Store the current profile as candidate (simple approach)
        self.candidate = list(self.profile)
        
        # In a more advanced implementation, you would run the train optimization
        # with the new desired profile and set the candidate accordingly
        
        # Calculate improvement
        e_m = np.linalg.norm(np.array(self.profile)-np.array(p_m)) - np.linalg.norm(np.array(self.candidate)-np.array(p_m))
        
        return e_m
        
    def accept(self):
        """Accept the candidate profile"""
        diff = list(map(operator.sub, self.candidate, self.profile))
        self.profile = list(self.candidate)
        return diff
    
    def apply_profile_to_model(self, profile):
        """Apply the profile steering suggested profile to the train model"""
        if self.train_model is None or self.data is None:
            return False
        
        # Convert time-based profile to distance-based constraints
        # This would involve adding new constraints to the train model
        # to follow the suggested power profile
        
        # Example (conceptual):
        # 1. Resample the profile to match distance points
        # 2. Add constraints to the model to follow this profile
        # 3. Re-solve the model with these additional constraints
        
        # This is a complex part that would require deeper integration with your model
        print("Applying profile to train model (would need implementation)")
        return True