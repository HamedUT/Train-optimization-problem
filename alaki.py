import numpy as np
import matplotlib.pyplot as plt

def generate_pv_profile(start_time_hour=8, duration_seconds=3600, max_capacity_mw=1.0, 
                       weather_condition="sunny", day_of_year=172, random_seed=None):
    """
    Generate a realistic PV generation profile based on time of day and weather conditions for the Netherlands.
    
    Args:
        start_time_hour (int or float): Hour of day when the simulation starts (0-23)
        duration_seconds (int): Duration of the simulation in seconds
        max_capacity_mw (float): Maximum capacity of the PV installation in MW
        weather_condition (str): Weather condition - "sunny", "partly_cloudy", "cloudy", "rainy"
        day_of_year (int): Day of year (1-365) for seasonal adjustments
        random_seed (int): Seed for reproducible randomization, None for random
        
    Returns:
        list: PV generation profile in MW for each second of the simulation
    """
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Determine seasonal factor (1.0 in summer, 0.4 in winter)
    summer_peak = 172  # June 21st
    winter_peak = 355  # December 21st
    days_from_winter = min((day_of_year - winter_peak) % 365, (winter_peak - day_of_year) % 365)
    seasonal_factor = 0.4 + 0.6 * (days_from_winter / 182.5)
    
    # Weather condition factors
    weather_factors = {
        "sunny": 1.0,
        "partly_cloudy": 0.7,
        "cloudy": 0.4,
        "rainy": 0.2
    }
    weather_factor = weather_factors.get(weather_condition.lower(), 0.7)
    
    # Generate time points for the simulation duration
    time_points = np.arange(0, duration_seconds)
    
    # Generate output profile
    pv_profile = []
    
    # Calculate sun parameters for the Netherlands
    sunrise_hour = 5.5 + (1 - seasonal_factor) * 2  # Earlier in summer (4-5 AM), later in winter (7-8 AM)
    sunset_hour = 21.0 - seasonal_factor * 3        # Later in summer (21-22 PM), earlier in winter (17-18 PM)
    daylight_hours = sunset_hour - sunrise_hour
    peak_hour = sunrise_hour + daylight_hours / 2   # Solar noon
    
    # Process each second of the simulation
    for second in time_points:
        # Convert simulation time to hour of day
        current_hour = (start_time_hour + second / 3600) % 24
        
        # Calculate solar output (bell curve based on time of day)
        if sunrise_hour <= current_hour <= sunset_hour:
            # Distance from solar noon normalized to a -1 to 1 range
            normalized_time = 2 * (current_hour - peak_hour) / daylight_hours
            
            # Bell curve formula: e^(-x²)
            sun_position_factor = np.exp(-(normalized_time ** 2) * 4)
            
            # Add some realistic variations for clouds/atmosphere
            variation = 1.0 + 0.1 * np.sin(second / 300) + 0.05 * np.sin(second / 120)
            
            # Add small random variations (1-minute cloud movements)
            if second % 60 == 0:
                cloud_variation = 0.95 + 0.1 * np.random.random()
            else:
                cloud_variation = 1.0
                
            # Calculate output with all factors combined
            output = max_capacity_mw * sun_position_factor * weather_factor * seasonal_factor * variation * cloud_variation
            
            # Ensure output doesn't exceed capacity or go negative
            output = max(0, min(output, max_capacity_mw))
            
            # Generate negative profile for consumption convention
            pv_profile.append(-output)
        else:
            # No generation during night
            pv_profile.append(0)
    
    return pv_profile

# Usage example in setup_profile_steering_environment:
generation_profile = generate_pv_profile(
    start_time_hour=8, 
    duration_seconds=60*60*8,
    max_capacity_mw=1.0,
    weather_condition="sunny",
    day_of_year=200)  # Example: Day 200 (July 19th)
plt.plot(generation_profile)  # Plotting the generated profile
plt.xlabel("Time (seconds)")
plt.ylabel("PV Generation (MW)")
plt.title("PV Generation Profile for the Netherlands")
plt.show()