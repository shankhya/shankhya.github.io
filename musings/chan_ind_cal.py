#Channel-independent calibration
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Example CIELAB values for medium and different digital levels
lab_medium = np.array([95, 0, 0])  # Example CIELAB values for bare medium
lab_values_cyan = np.array([
    [90, -5, -5],
    [80, -10, -10],
    [70, -15, -15],
    [60, -20, -20],
    [50, -25, -25]
])
lab_values_magenta = np.array([
    [90, 5, -5],
    [80, 10, -10],
    [70, 15, -15],
    [60, 20, -20],
    [50, 25, -25]
])
lab_values_yellow = np.array([
    [90, 0, 5],
    [80, 0, 10],
    [70, 0, 15],
    [60, 0, 20],
    [50, 0, 25]
])

# Digital levels corresponding to the CIELAB values
digital_levels = np.array([0, 64, 128, 192, 255])

# Step 3: Calculate Mi(d) using ΔE*ab color difference for each color channel
def delta_e_ab(lab1, lab2):
    return np.sqrt(np.sum((lab1 - lab2)**2))

# Calculate Mi(d) for cyan, magenta, and yellow
Mi_cyan_raw = [delta_e_ab(lab_medium, lab) for lab in lab_values_cyan]
Mi_magenta_raw = [delta_e_ab(lab_medium, lab) for lab in lab_values_magenta]
Mi_yellow_raw = [delta_e_ab(lab_medium, lab) for lab in lab_values_yellow]

# Step 4: Scale Mi(d) such that Mi(d_max) = d_max
d_max = 255
Mi_cyan_scaled = np.array(Mi_cyan_raw) * (d_max / Mi_cyan_raw[-1])
Mi_magenta_scaled = np.array(Mi_magenta_raw) * (d_max / Mi_magenta_raw[-1])
Mi_yellow_scaled = np.array(Mi_yellow_raw) * (d_max / Mi_yellow_raw[-1])

# Step 5: Invert Mi(d) using interpolation to get Mi^{-1} (optional for other tasks)
cyan_interp = interp1d(Mi_cyan_scaled, digital_levels, kind='linear', fill_value="extrapolate")
magenta_interp = interp1d(Mi_magenta_scaled, digital_levels, kind='linear', fill_value="extrapolate")
yellow_interp = interp1d(Mi_yellow_scaled, digital_levels, kind='linear', fill_value="extrapolate")

# Step 6: Plot the calibration curves for unscaled and scaled Mi(d)
plt.figure(figsize=(12, 6))

# Plot (a): Unscaled Raw Device Response Mi(d)
plt.subplot(1, 2, 1)
plt.plot(digital_levels, Mi_cyan_raw, label='Cyan Raw Mi(d)')
plt.plot(digital_levels, Mi_magenta_raw, label='Magenta Raw Mi(d)')
plt.plot(digital_levels, Mi_yellow_raw, label='Yellow Raw Mi(d)')
plt.xlabel('Digital Level')
plt.ylabel('Mi(d) (ΔE*ab)')
plt.title('Raw Device Response Mi(d)')
plt.legend()
plt.grid(True)

# Plot (b): Scaled Mi(d) (Calibration Curves)
plt.subplot(1, 2, 2)
plt.plot(digital_levels, Mi_cyan_scaled, label='Cyan Scaled Mi(d)')
plt.plot(digital_levels, Mi_magenta_scaled, label='Magenta Scaled Mi(d)')
plt.plot(digital_levels, Mi_yellow_scaled, label='Yellow Scaled Mi(d)')
plt.xlabel('Digital Level')
plt.ylabel('Mi(d) (ΔE*ab)')
plt.title('Scaled Calibration Curves Mi(d)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


