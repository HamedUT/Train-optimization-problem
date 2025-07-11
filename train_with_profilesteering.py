import sys
sys.path.append('/profilesteering')

import time
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyomo
from Train import main as train_main
from Train import Main, process_speed_limits, collecting_gradients
from Train import electrical, train, simulation
from train_device import TrainDevice
from train_battery import TrainBattery
from profilesteering.profilesteering import ProfileSteering
from pv_device import PVDevice

try:
    from profilesteering.profilesteering import ProfileSteering
    print("Successfully imported ProfileSteering")
except ImportError:
    print("ERROR: ProfileSteering package not found!")
    print("Please check your path settings")
    sys.exit(1)

def run_integrated_optimization():
    """Run the integrated train optimization with profile steering"""
    # Initialize the train model first
    print("Initializing train model...")
    speed_limits_dict, speed_limit_array, simulation.distance_remaining = process_speed_limits(simulation)
    gradients = collecting_gradients(simulation, mode="const", max_gradient=0.0)
    
    # Build data dictionary
    data = {0: {'grade': gradients[0], 'speed_limit': speed_limit_array[0]}}
    for i in range(1, int(simulation.distance_remaining / simulation.delta_s) + 1):
        data[i * simulation.delta_s] = {
            'grade': gradients[min(i, len(gradients) - 1)],
            'speed_limit': speed_limit_array[min(i, len(speed_limit_array) - 1)]
        }
    
    # First run: Get initial solution from train model
    print("Running initial train optimization...")
    model, termination_condition = Main(data, train, electrical, simulation)
    
    if termination_condition not in [pyomo.TerminationCondition.optimal, pyomo.TerminationCondition.locallyOptimal]:
        print("Initial train optimization failed. Exiting.")
        return
    
    # Extract total simulation time for profile creation
    final_distance = list(data.keys())[-1]
    total_time = model.t[final_distance]()
    print(f"Train simulation complete. Total time: {total_time:.2f} seconds")
    
    # Create time-based discretization for profile steering
    time_intervals = 100  # Adjust as needed
    
    # Initialize device instances
    train_device = TrainDevice(model, train, electrical, simulation)
    train_device.set_data(data)
    
    train_battery = TrainBattery(model, train, electrical)
    train_battery.set_data(data)
    
    # Initialize PV device with 1.5MW peak power
    pv_device = PVDevice(peak_power=1.5e6)
    
    # Initialize profile steering with all devices
    devices = [train_device, train_battery, pv_device]
    ps = ProfileSteering(devices)
    
    # Create a desired profile (flat, minimal energy consumption)
    desired_profile = [0] * time_intervals
    
    # Initialize profiles
    print("Initializing profiles...")
    initial_power_profile = ps.init(desired_profile)
    
    # Run the profile steering algorithm
    print("Running Profile Steering optimization...")
    e_min = 0.001  # Minimum improvement threshold
    max_iters = 20  # Maximum iterations (increased for better convergence)
    
    final_profile = ps.iterative(e_min, max_iters)
    
    # Visualize results
    plot_profiles(desired_profile, initial_power_profile, final_profile, 
                  train_device.profile, pv_device.profile, train_battery.profile)
    
    print("Optimization complete.")
    
    analyze_battery_charging(train_device.profile, pv_device.profile, train_battery.profile)

def plot_profiles(desired, initial, final, train_profile, pv_profile, battery_profile):
    """Plot the desired, initial and final profiles along with individual device profiles and battery SoC"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot aggregated profiles on first subplot
    ax1.plot(desired, 'g-', label='Desired Profile', linewidth=2)
    ax1.plot(initial, 'b-', label='Initial Aggregate Profile', linewidth=2)
    ax1.plot(final, 'r-', label='Optimized Aggregate Profile', linewidth=2)
    ax1.set_ylabel('Power (W)', fontsize=12)
    ax1.set_title('ProfileSteering Aggregated Power Profiles', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Plot individual device profiles on second subplot
    ax2.plot(train_profile, 'b-', label='Train Power', linewidth=2)
    ax2.plot(pv_profile, 'y-', label='PV Generation', linewidth=2)
    ax2.plot(battery_profile, 'g-', label='Battery Power', linewidth=2)
    ax2.plot([0] * len(train_profile), 'k--', linewidth=1)  # Zero line
    ax2.set_ylabel('Power (W)', fontsize=12)
    ax2.set_title('Individual Device Power Profiles', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Plot battery SoC on third subplot
    initial_soc = electrical.batt_cap * electrical.batt_initial_soc  # Initial SoC in Wh
    time_step = simulation.time_remaining / len(battery_profile)  # Time step in seconds
    soc_values = [initial_soc]
    
    # Calculate SoC for each time step
    # Negative battery power = charging, positive = discharging
    for i in range(len(battery_profile)):
        # Convert power (W) to energy (Wh) for this time step
        energy_change = -battery_profile[i] * (time_step / 3600)  # Negative of battery power because negative = charging
        new_soc = soc_values[-1] + energy_change
        # Constrain SoC to battery capacity limits
        new_soc = max(0, min(new_soc, electrical.batt_cap))
        soc_values.append(new_soc)
    
    # Remove first point (initial SoC) to match length of other arrays
    soc_values = soc_values[1:]
    
    # Convert Wh to kWh for better readability
    soc_values_kwh = [soc / 1000 for soc in soc_values]
    
    ax3.plot(soc_values_kwh, 'm-', linewidth=2)
    ax3.set_xlabel('Time Interval', fontsize=12)
    ax3.set_ylabel('Battery SoC (kWh)', fontsize=12)
    ax3.set_title('Battery State of Charge', fontsize=14)
    ax3.grid(True)
    
    # Add horizontal lines for min/max SoC
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=electrical.batt_cap/1000, color='r', linestyle='--', alpha=0.5)
    ax3.text(0, electrical.batt_cap/1000 * 1.02, f'Max: {electrical.batt_cap/1000:.2f} kWh', 
             verticalalignment='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('profile_steering_with_pv_and_soc.png', dpi=300)
    plt.show()

def analyze_battery_charging(train_profile, pv_profile, battery_profile):
    """Analyze when the battery charges from regenerative braking vs PV generation"""
    regen_charging_intervals = 0
    pv_charging_intervals = 0
    total_charging_intervals = 0
    
    regen_energy = 0
    pv_energy = 0
    total_charging_energy = 0
    
    time_step = simulation.time_remaining / len(battery_profile)  # Time step in seconds
    
    for i in range(len(battery_profile)):
        # Battery charging happens when battery power is negative
        if battery_profile[i] < 0:
            total_charging_intervals += 1
            total_charging_energy += -battery_profile[i] * (time_step / 3600)  # Wh
            
            # Regenerative braking happens when train power is negative
            if train_profile[i] < 0:
                regen_charging_intervals += 1
                regen_energy += -battery_profile[i] * (time_step / 3600)  # Wh
            
            # PV charging happens when PV generation is non-zero (negative values)
            if pv_profile[i] < 0:
                pv_charging_intervals += 1
                pv_energy += -battery_profile[i] * (time_step / 3600)  # Wh
    
    print("\nBattery Charging Analysis:")
    print(f"Total charging intervals: {total_charging_intervals} out of {len(battery_profile)}")
    print(f"Charging from regenerative braking: {regen_charging_intervals} intervals ({regen_energy:.2f} Wh)")
    print(f"Charging during PV generation: {pv_charging_intervals} intervals ({pv_energy:.2f} Wh)")
    print(f"Total charging energy: {total_charging_energy:.2f} Wh")
    
    # Detect overlap between regen and PV charging
    if regen_charging_intervals + pv_charging_intervals > total_charging_intervals:
        overlap = regen_charging_intervals + pv_charging_intervals - total_charging_intervals
        print(f"Note: {overlap} intervals had both regenerative braking and PV generation")

if __name__ == "__main__":
    run_integrated_optimization()