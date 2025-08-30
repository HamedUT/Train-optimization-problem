"""
Peak Shaving Visualization Tool

This script visualizes the results of peak power demand shaving for train operations.
It compares the initial power profile with the final optimized profile after peak shaving
has been applied, highlighting the reduction in maximum power demand.

The script:
1. Locates the most recent peak shaving data CSV file in the output_data directory
2. Reads the time series data containing power profiles
3. Creates a comparison plot between the initial and final power profiles
4. Calculates and displays the peak reduction (in MW and percentage)
5. Saves the visualization as a high-resolution PNG image

This visualization helps engineers and operators analyze how effectively the peak shaving
strategies have reduced maximum power demands from the electrical grid, which can lead
to significant cost savings and improved grid stability.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Find the most recent CSV file in the output_data directory
output_dir = "output_data"
csv_files = [f for f in os.listdir(output_dir) if f.startswith("peak_shaving_profiles_") and f.endswith(".csv")]

if not csv_files:
    print("Error: No peak_shaving_profiles CSV files found in the output_data directory.")
    exit()

# Sort files by modification time (most recent last)
csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))

# Select the most recent file
csv_file_path = os.path.join(output_dir, csv_files[-1])
print(f"Using the most recent CSV file: {csv_file_path}")

try:
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Check CSV structure
    print(f"CSV columns: {df.columns.tolist()}")
    
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    
    # Plot time vs each profile
    time_column = 'Time(s)'
    
    # Get the profile columns (excluding time column)
    profile_columns = [col for col in df.columns if col != time_column]
    
    # Plot only first and last profile columns
    if profile_columns:
        first_column = profile_columns[0]
        last_column = profile_columns[-1]
        
        # Plot initial profile
        plt.plot(df[time_column], df[first_column], label=first_column, color='blue', linewidth=2)
        
        # Plot final profile
        plt.plot(df[time_column], df[last_column], label=last_column, color='red', linewidth=2)
        
        # Annotate the peaks
        initial_peak_idx = df[first_column].idxmax()
        initial_peak = df[first_column].max()
        initial_peak_time = df[time_column][initial_peak_idx]
        
        final_peak_idx = df[last_column].idxmax()
        final_peak = df[last_column].max()
        final_peak_time = df[time_column][final_peak_idx]
        
        # Mark the peaks
        plt.plot(initial_peak_time, initial_peak, 'o', color='blue', markersize=8)
        plt.plot(final_peak_time, final_peak, 'o', color='red', markersize=8)
        
        # Calculate peak reduction
        reduction = initial_peak - final_peak
        reduction_pct = (reduction / initial_peak) * 100
    
    # Customize the plot
    plt.xlabel('Time (seconds)')
    plt.ylabel('Power (MW)')
    plt.title(f'Peak Shaving: Initial vs Final Profile\nPeak Reduction: {reduction:.3f} MW ({reduction_pct:.1f}%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    plt.savefig('peak_shaving_comparison.png', dpi=600, bbox_inches='tight')
    
    # Display the plot
    plt.show()
    
except Exception as e:
    print(f"Error plotting data: {e}")
    print("Please check the CSV structure and adjust the column names in the script.")
    # Print the first few rows of the CSV for debugging
    try:
        df_debug = pd.read_csv(csv_file_path)
        print("\nCSV file content (first 5 rows):")
        print(df_debug.head())
    except Exception as debug_e:
        print(f"Couldn't read CSV file for debugging: {debug_e}")