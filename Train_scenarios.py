"""
Train Scenarios Simulation Framework

This script runs multiple train optimization simulations with varying parameters to analyze
the impact of different train and operational characteristics on energy consumption and performance.

The script allows you to:
1. Define ranges for various train parameters (mass, efficiency, acceleration, braking, etc.)
2. Iterate through all combinations of these parameters
3. Run the train optimization model for each parameter set
4. Save the results to a CSV file for further analysis
5. Track progress and timing information during execution

Parameters that can be varied:
- m: Train mass (kg)
- eta: Propulsion efficiency
- braking_eff: Regenerative braking efficiency
- max_acc: Maximum acceleration (m/s²)
- max_braking: Maximum braking deceleration (m/s²)
- max_p: Maximum power (W)
- total_time: Journey time (seconds)
- delta_s: Distance step size (m)
- speed_limit_csv_path: Path to CSV file containing speed limits

Output:
- CSV file with all parameter combinations and their resulting energy consumption and maximum velocity
- Progress updates during execution
- Statistics on feasible vs. infeasible solutions
"""

import sys
import csv
import time
import itertools
import pyomo.environ as pyomo
from Train import (
    TrainParams, ElectricalParams, SimulationParams,
    process_speed_limits, collecting_gradients, Main, calculate_energy_consumption
)

def run_simulation(params_dict):
    """Run a single simulation with the given parameters and return results"""
    # Create parameter objects with updated values
    train = TrainParams(
        m=params_dict['m'],
        eta=params_dict['eta'],
        braking_eff=params_dict['braking_eff'],
        max_acc=params_dict['max_acc'],
        max_braking=params_dict['max_braking'],
        max_p=params_dict['max_p']
    )
    
    electrical = ElectricalParams()
    
    simulation = SimulationParams(
        total_time=params_dict['total_time'],
        delta_s=params_dict['delta_s']
    )
    # Set speed_limit_csv_path after initialization
    simulation.speed_limit_csv_path = params_dict['speed_limit_csv_path']
    simulation.time_remaining = simulation.total_time - simulation.t_init
    
    # Process speed limits and prepare data
    speed_limits_dict, speed_limit_array, simulation.distance_remaining = process_speed_limits(simulation)
    gradients = collecting_gradients(simulation, mode="const", max_gradient=0.0)
    
    # Prepare data structure
    data = {0: {'grade': gradients[0], 'speed_limit': speed_limit_array[0]}}
    for i in range(1, int(simulation.distance_remaining / simulation.delta_s) + 1):
        data[i * simulation.delta_s] = {
            'grade': gradients[min(i, len(gradients) - 1)],
            'speed_limit': speed_limit_array[min(i, len(speed_limit_array) - 1)]
        }
    
    # Run the optimization model
    model, termination_condition = Main(data, train, electrical, simulation)
    
    if termination_condition in [pyomo.TerminationCondition.optimal, pyomo.TerminationCondition.locallyOptimal]:
        # Calculate energy consumption
        energy_consumption = calculate_energy_consumption(model, data, simulation, print_results=False)
        
        # Find maximum velocity
        max_velocity = 0
        for d in data.keys():
            velocity = model.v[d]() * 3.6  # Convert to km/h
            max_velocity = max(max_velocity, velocity)
        
        return {
            "Train_energy_consumption": energy_consumption,
            "max_velocity": max_velocity,
            "feasible": True
        }
    else:
        return {"feasible": False}

def main():
    # Define parameter ranges to test
    param_ranges = {
        'm': [390000],                  # kg
        'eta': [0.85, 0.9, 0.95],                       # Propulsion efficiency
        'braking_eff': [0.7, 0.8, 0.9],                 # Regenerative braking efficiency
        'max_acc': [0.5, 0.7, 0.9],                     # m/s^2
        'max_braking': [0.3, 0.5, 0.7],                 # m/s^2
        'max_p': [1500000, 2000000, 2500000],           # W
        'total_time': [360, 420, 480],                  # sec
        'delta_s': [100],                      # m
        'speed_limit_csv_path': [r"c:\Users\DarbandiH\OneDrive - University of Twente\Postdoc\Python\Train optimization problem\SpeedLimits\SpeedLimit_Alaki08.csv",
                                 r"c:\Users\DarbandiH\OneDrive - University of Twente\Postdoc\Python\Train optimization problem\SpeedLimits\SpeedLimit_Alaki10.csv",
                                 r"c:\Users\DarbandiH\OneDrive - University of Twente\Postdoc\Python\Train optimization problem\SpeedLimits\SpeedLimit_Alaki12.csv"]
    }
    
    # Prepare CSV file
    csv_file = "train_simulation_results.csv"
    
    # Create header for the CSV file
    header = list(param_ranges.keys()) + ["Train_energy_consumption", "max_velocity"]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        
        # Get all parameter combinations
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        # Track progress
        total_combinations = 1
        for val_list in values:
            total_combinations *= len(val_list)
        
        print(f"Total parameter combinations to test: {total_combinations}")
        counter = 0
        feasible_counter = 0
        start_time = time.time()
        
        # Iterate through all parameter combinations
        for combination in itertools.product(*values):
            counter += 1
            params_dict = {keys[i]: combination[i] for i in range(len(keys))}
            
            print(f"Testing combination {counter}/{total_combinations}: {params_dict}")
            
            try:
                results = run_simulation(params_dict)
                
                if results["feasible"]:
                    # Combine parameters and results
                    row = params_dict.copy()
                    
                    # Extract just the two characters before .csv from the speed limit path
                    full_path = row['speed_limit_csv_path']
                    filename = full_path.split('\\')[-1]  # Get the filename
                    short_id = filename[-6:-4]  # Extract the two chars before .csv
                    row['speed_limit_csv_path'] = int(short_id)
                    
                    row["Train_energy_consumption"] = round(results["Train_energy_consumption"], 3)
                    row["max_velocity"] = round(results["max_velocity"], 2)
                    
                    # Write to CSV
                    writer.writerow(row)
                    feasible_counter += 1
                    print(f"✓ Feasible solution found: Energy={results['Train_energy_consumption']:.2f} kWh, Max Speed={results['max_velocity']:.2f} km/h")
                else:
                    print("✗ Infeasible solution - skipping")
                
            except Exception as e:
                print(f"✗ Error occurred: {str(e)} - skipping")
            
            # Show progress
            elapsed = time.time() - start_time
            avg_time = elapsed / counter
            remaining = avg_time * (total_combinations - counter)
            print(f"Progress: {counter}/{total_combinations} ({counter/total_combinations*100:.1f}%)")
            print(f"Time elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")
            print(f"Feasible solutions: {feasible_counter}/{counter} ({feasible_counter/counter*100 if counter > 0 else 0:.1f}%)")
            print("-" * 80)
    
    print(f"Simulation complete! Results saved to {csv_file}")
    print(f"Total feasible solutions: {feasible_counter}/{total_combinations} ({feasible_counter/total_combinations*100:.1f}%)")

if __name__ == "__main__":
    main()
