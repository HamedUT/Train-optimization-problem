import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def plot_train_problem(rho_1, rho_2, S, V0):
    """Visualizes the train problem setup with more details."""

    fig, ax = plt.subplots(figsize=(12, 7))  # Slightly larger figure

    # 1. Track and Ground
    ax.plot([0, S], [0, 0], color='gray', linewidth=3, label="Track")
    ax.fill_between([0, S], [-0.2, -0.2], color='lightgray')  # Ground

    # 2. Substations (with voltage representation)
    substation_height = 0.5
    ax.plot([0, S], [substation_height, substation_height], '^', color='blue', markersize=12, label="Substations")
    ax.annotate("Substation 1 (V0)", (0, substation_height + 0.1), textcoords="offset points", xytext=(0,10), ha='center')
    ax.annotate("Substation 2", (S, substation_height + 0.1), textcoords="offset points", xytext=(0,10), ha='center')

    # Voltage representation (more visual)
    voltage_line_y = substation_height * 0.8
    ax.plot([0, S], [voltage_line_y, voltage_line_y], linestyle='--', color='orange', linewidth=2, label="Voltage Lines")
    ax.annotate("V0", (0, voltage_line_y + 0.05), color="orange")  # Label V0 only at the start

    # 3. Train (realistic representation)
    train_x = S / 4  # Example position (you can change this)
    train_y = 0.15
    train_width = S * 0.05  # Adjust train size relative to track length
    train_height = 0.15
    rect = patches.Rectangle((train_x - train_width / 2, train_y - train_height/2), train_width, train_height, linewidth=1, edgecolor='red', facecolor='red', label="Train")
    ax.add_patch(rect)


    # 4. Resistors (visual representation)
    resistor_y = voltage_line_y * 0.6
    ax.plot([0, train_x], [resistor_y, resistor_y], color='black', linewidth=2)
    ax.plot(train_x, resistor_y, marker="$\mathrm{R_1}$", markersize=15, color='black')  # Resistor symbol
    ax.annotate(r"$\rho_1$", (train_x/2, resistor_y - 0.1), ha='center')

    ax.plot([train_x, S], [resistor_y, resistor_y], color='black', linewidth=2)
    ax.plot(train_x, resistor_y, marker="$\mathrm{R_2}$", markersize=15, color='black') # Resistor symbol
    ax.annotate(r"$\rho_2$", ((train_x+S)/2, resistor_y - 0.1), ha='center')


    # 5. Annotations and Labels
    ax.set_xlabel("Distance (m)", fontsize=14)
    ax.set_ylabel("", fontsize=14)  # No meaningful y-axis
    ax.set_title("Train Energy Optimization Problem", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12) # Larger tick labels

    # 6. Legend
    ax.legend(fontsize=12, loc="upper right")

    # 7. Axis Limits and Grid
    ax.set_xlim(-S*0.05, S*1.05)
    ax.set_ylim(-0.3, substation_height + 0.3)  # Adjust y-limits
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# Example usage (same as before)
rho_1 = 0.00003  # Ohms/m
rho_2 = 0.00003  # Ohms/m
S = 100  # m
V0 = 1500  # V

plot_train_problem(rho_1, rho_2, S, V0)