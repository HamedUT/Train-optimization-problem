import matplotlib.pyplot as plt
import pandas as pd
import os


def extract_total_energy(filepath):
    """Extracts the total energy value from the last non-empty line of the CSV file."""
    with open(filepath, 'r', encoding='latin1') as f:
        lines = [line.strip() for line in f if line.strip()]
        for line in reversed(lines):
            if line.startswith("Total Energy Consumption"):
                # Handles both comma and colon as separator
                parts = line.replace(':', ',').split(',')
                for part in parts:
                    try:
                        return float(part)
                    except ValueError:
                        continue
    return None

def read_clean_csv(filepath):
    """Reads only the data rows from the CSV, skipping summary lines at the end."""
    with open(filepath, 'r', encoding='latin1') as f:
        lines = f.readlines()
    # Keep only lines that have 4 columns (data lines)
    data_lines = [line for line in lines if line.count(',') == 3 and not line.startswith("Distance")]
    # Add header back
    header = "Distance (km),Velocity (km/h),Power (MW),Acceleration (m/s²)\n"
    clean_csv = header + ''.join(data_lines)
    from io import StringIO
    return pd.read_csv(StringIO(clean_csv))

def plot_velocity_comparison(file1, file2, file3, file4, distance, time, RB):
    te1 = extract_total_energy(file1)
    te2 = extract_total_energy(file2)
    te3 = extract_total_energy(file3)
    te4 = extract_total_energy(file4)

    data1 = read_clean_csv(file1)
    data2 = read_clean_csv(file2)
    data3 = read_clean_csv(file3)
    data4 = read_clean_csv(file4)

    distances1 = data1['Distance (km)']
    velocities1 = data1['Velocity (km/h)']
    distances2 = data2['Distance (km)']
    velocities2 = data2['Velocity (km/h)']
    distances3 = data3['Distance (km)']
    velocities3 = data3['Velocity (km/h)']
    distances4 = data4['Distance (km)']
    velocities4 = data4['Velocity (km/h)']

    plt.figure(figsize=(12, 6))
    plt.plot(distances1, velocities1, '--', color='red', label=f'Velocity (RMS) [{te1:.2f} kWh]', linewidth=2)
    plt.plot(distances2, velocities2, '--', color='orange', label=f'Velocity (Model) [{te2:.2f} kWh]', linewidth=2)
    plt.plot(distances3, velocities3, '--', color='green', label=f'Velocity (MC) [{te3:.2f} kWh]', linewidth=2)
    plt.plot(distances4, velocities4, '--', color='purple', label=f'Velocity (EETC) [{te4:.2f} kWh]', linewidth=2)
    plt.xlabel('Distance (km)')
    plt.ylabel('Velocity (km/h)')
    plt.title('Velocity Comparison')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    folder = "Strategies comparison plots"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"velocity_comparison_{distance}km_{time}s_{RB}RB.png")
    plt.savefig(filename, dpi=600, bbox_inches='tight', transparent=True)

def plot_power_comparison(file1, file2, file3, file4, distance, time, RB):
    te1 = extract_total_energy(file1)
    te2 = extract_total_energy(file2)
    te3 = extract_total_energy(file3)
    te4 = extract_total_energy(file4)

    data1 = read_clean_csv(file1)
    data2 = read_clean_csv(file2)
    data3 = read_clean_csv(file3)
    data4 = read_clean_csv(file4)

    distances1 = data1['Distance (km)']
    powers1 = data1['Power (MW)']
    distances2 = data2['Distance (km)']
    powers2 = data2['Power (MW)']
    distances3 = data3['Distance (km)']
    powers3 = data3['Power (MW)']
    distances4 = data4['Distance (km)']
    powers4 = data4['Power (MW)']

    plt.figure(figsize=(12, 6))
    plt.plot(distances2, powers2, '-', color='orange', label=f'Power (Model) [{te2:.2f} kWh]', linewidth=2)
    plt.plot(distances1, powers1, '-', color='red', label=f'Power (RMS) [{te1:.2f} kWh]', linewidth=2)
    plt.plot(distances3, powers3, '-', color='green', label=f'Power (MC) [{te3:.2f} kWh]', linewidth=2)
    plt.plot(distances4, powers4, '-', color='purple', label=f'Power (EETC) [{te4:.2f} kWh]', linewidth=2)
    plt.xlabel('Distance (km)')
    plt.ylabel('Power (MW)')
    plt.title('Power Comparison')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    folder = "Strategies comparison plots"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"power_comparison_{distance}km_{time}s_{RB}RB.png")
    plt.savefig(filename, dpi=600, bbox_inches='tight', transparent=True)
def plot_velocity_and_power_combined(file1, file2, file3, file4, distance, time, RB):
    data1 = read_clean_csv(file1)
    data2 = read_clean_csv(file2)
    data3 = read_clean_csv(file3)
    data4 = read_clean_csv(file4)

    distances1 = data1['Distance (km)']
    velocities1 = data1['Velocity (km/h)']
    powers1 = data1['Power (MW)']
    distances2 = data2['Distance (km)']
    velocities2 = data2['Velocity (km/h)']
    powers2 = data2['Power (MW)']
    distances3 = data3['Distance (km)']
    velocities3 = data3['Velocity (km/h)']
    powers3 = data3['Power (MW)']
    distances4 = data4['Distance (km)']
    velocities4 = data4['Velocity (km/h)']
    powers4 = data4['Power (MW)']

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot velocities
    ax1.plot(distances1, velocities1, '--', color='red', label='Velocity (RMS)', linewidth=2)
    ax1.plot(distances2, velocities2, '--', color='orange', label='Velocity (Model)', linewidth=2)
    ax1.plot(distances3, velocities3, '--', color='green', label='Velocity (MC)', linewidth=2)
    ax1.plot(distances4, velocities4, '--', color='purple', label='Velocity (EETC)', linewidth=2)
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Velocity (km/h)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Plot powers on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(distances1, powers1, '-', color='red', label='Power (RMS)', linewidth=1)
    ax2.plot(distances2, powers2, '-', color='orange', label='Power (Model)', linewidth=1)
    ax2.plot(distances3, powers3, '-', color='green', label='Power (MC)', linewidth=1)
    ax2.plot(distances4, powers4, '-', color='purple', label='Power (EETC)', linewidth=1)
    ax2.set_ylabel('Power (MW)', color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Velocity and Power Comparison')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

# Example usage
if __name__ == "__main__":
    file1 = 'RMS_results.csv'
    file2 = 'Train_results.csv'
    file3 = 'MC_results.csv'
    file4 = 'EETC_results.csv'

    distance = input("Enter the distance (in km): ")
    time = input("Enter the time (in s): ")
    RB = input("Enter the regenerative braking efficiency: ")

    plot_velocity_comparison(file1, file2, file3, file4, distance, time, RB)
    plot_power_comparison(file1, file2, file3, file4, distance, time, RB)
    plot_velocity_and_power_combined(file1, file2, file3, file4, distance, time, RB)
    plt.show()
