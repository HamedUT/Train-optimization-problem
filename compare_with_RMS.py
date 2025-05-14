import matplotlib.pyplot as plt
import pandas as pd

def plot_velocity_and_power_comparison(file1, file2):
    """
    Plots velocity and power outputs from two CSV files for comparison.
    """
    # Read data from CSV files with explicit encoding
    data1 = pd.read_csv(file1, encoding='latin1')  # Change 'latin1' if needed
    data2 = pd.read_csv(file2, encoding='latin1')

    # Extract columns
    distances1 = data1['Distance (km)']  # Ensure column names match the CSV files
    velocities1 = data1['Velocity (km/h)']
    powers1 = data1['Power (MW)']

    distances2 = data2['Distance (km)']
    velocities2 = data2['Velocity (km/h)']
    powers2 = data2['Power (MW)']

    # Plot data
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot velocity on the primary y-axis
    ax1.plot(distances1, velocities1, 'r-', label='Velocity (RMS)', linewidth=2)
    ax1.plot(distances2, velocities2, 'r--', label='Velocity (Model)', linewidth=2)
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Velocity (km/h)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    # Plot power on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(distances1, powers1, 'b-', label='Power (RMS)', linewidth=2)
    ax2.plot(distances2, powers2, 'b--', label='Power (Model)', linewidth=2)
    ax2.set_ylabel('Power (MW)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Velocity and Power Comparison')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace these with the actual paths to your CSV files
    file1 = 'velocity_power_results.csv'
    file2 = 'power_velocity_acceleration_results.csv'

    plot_velocity_and_power_comparison(file1, file2)
