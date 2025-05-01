import matplotlib.pyplot as plt
import pandas as pd

# Function to read power values from first line
def read_power_values(filename):
    with open(filename, 'r') as file:
        # Read first two lines
        power_line = file.readline()
        efficiency_line = file.readline()
        
        # Extract power values
        values = power_line.split(',')
        P_sub1 = float(values[0].split(':')[1].replace('W', '').strip())
        P_sub2 = float(values[1].split(':')[1].replace('W', '').strip())
        
        # Extract efficiency value from "Braking Efficiency: XX.X%" format
        eff = float(efficiency_line.split('Braking Efficiency:')[1].replace('%', '').strip())
        
        return P_sub1, P_sub2, eff

# Read data from files
file1_path = 't600_S10000_e0.txt'
file2_path = 't600_S10000_e1.txt'
file3_path = 't600_S10000_e2.txt'
file4_path = 't600_S10000_e3.txt'

# Read power values for all four files
P_sub1_max1, P_sub2_max1, eff1 = read_power_values(file1_path)
P_sub1_max2, P_sub2_max2, eff2 = read_power_values(file2_path)
P_sub1_max3, P_sub2_max3, eff3 = read_power_values(file3_path)
P_sub1_max4, P_sub2_max4, eff4 = read_power_values(file4_path)

# Load data using pandas, skip the first line
data1 = pd.read_csv(file1_path, header=2, names=['Time', 'Velocity', 'Power'])
data2 = pd.read_csv(file2_path, header=2, names=['Time', 'Velocity', 'Power'])
data3 = pd.read_csv(file3_path, header=2, names=['Time', 'Velocity', 'Power'])
data4 = pd.read_csv(file4_path, header=2, names=['Time', 'Velocity', 'Power'])

# Create velocity plot
plt.figure(figsize=(12, 7))
plt.plot(data1['Time'], data1['Velocity'], 
         label=f'e0 (P_sub1_max: {P_sub1_max1/1e6:.3f} MW, P_sub2_max: {P_sub2_max1/1e6:.3f} MW, Eff: {eff1:.0f}%)', 
         linewidth=2)
plt.plot(data2['Time'], data2['Velocity'], 
         label=f'e1 (P_sub1_max: {P_sub1_max2/1e6:.3f} MW, P_sub2_max: {P_sub2_max2/1e6:.3f} MW, Eff: {eff2:.0f}%)', 
         linewidth=2)
plt.plot(data3['Time'], data3['Velocity'], 
         label=f'e2 (P_sub1_max: {P_sub1_max3/1e6:.3f} MW, P_sub2_max: {P_sub2_max3/1e6:.3f} MW, Eff: {eff3:.0f}%)', 
         linewidth=2)
plt.plot(data4['Time'], data4['Velocity'], 
         label=f'e3 (P_sub1_max: {P_sub1_max4/1e6:.3f} MW, P_sub2_max: {P_sub2_max4/1e6:.3f} MW, Eff: {eff4:.0f}%)', 
         linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Velocity (km/h)')
plt.title('Velocity Comparison')
plt.legend()
plt.grid(True, axis='y')

# Create power plot
plt.figure(figsize=(12, 7))
plt.plot(data1['Time'], data1['Power'], 
         label=f'e0 (P_sub1_max: {P_sub1_max1/1e6:.3f} MW, P_sub2_max: {P_sub2_max1/1e6:.3f} MW, Eff: {eff1:.0f}%)', 
         linewidth=2)
plt.plot(data2['Time'], data2['Power'], 
         label=f'e1 (P_sub1_max: {P_sub1_max2/1e6:.3f} MW, P_sub2_max: {P_sub2_max2/1e6:.3f} MW, Eff: {eff2:.0f}%)', 
         linewidth=2)
plt.plot(data3['Time'], data3['Power'], 
         label=f'e2 (P_sub1_max: {P_sub1_max3/1e6:.3f} MW, P_sub2_max: {P_sub2_max3/1e6:.3f} MW, Eff: {eff3:.0f}%)', 
         linewidth=2)
plt.plot(data4['Time'], data4['Power'], 
         label=f'e3 (P_sub1_max: {P_sub1_max4/1e6:.3f} MW, P_sub2_max: {P_sub2_max4/1e6:.3f} MW, Eff: {eff4:.0f}%)', 
         linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Power (MW)')
plt.title('Power Comparison')
plt.legend()
plt.grid(True, axis='y')

plt.show()