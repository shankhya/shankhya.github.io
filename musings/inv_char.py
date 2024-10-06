#Inverse char
import numpy as np
from scipy.interpolate import interp1d

# Function to derive inverse CMYK values from target CIELAB
def inverse_cmyk_characterization(lab_values, cmyk_values, target_lab):
    # Interpolation for each CMYK channel
    C_interp_func = interp1d(lab_values[:, 0], cmyk_values[:, 0], bounds_error=False, fill_value='extrapolate')
    M_interp_func = interp1d(lab_values[:, 0], cmyk_values[:, 1], bounds_error=False, fill_value='extrapolate')
    Y_interp_func = interp1d(lab_values[:, 0], cmyk_values[:, 2], bounds_error=False, fill_value='extrapolate')
    K_interp_func = interp1d(lab_values[:, 0], cmyk_values[:, 3], bounds_error=False, fill_value='extrapolate')

    # Get the interpolated CMYK values for the target CIELAB values
    c = C_interp_func(target_lab[0])
    m = M_interp_func(target_lab[0])
    y = Y_interp_func(target_lab[0])
    k = K_interp_func(target_lab[0])

    return np.array([c, m, y, k])

# Measured color data (known input-output pairs for training)
# Expanded CIELAB values (L*, a*, b*) and corresponding CMYK values for training
lab_values = np.array([
    [50, 10, 10],
    [60, 20, 20],
    [70, 30, 30],
    [80, 40, 40],
    [90, 50, 50],
    [55, 15, 15],
    [65, 25, 25],
    [75, 35, 35],
    [85, 45, 45],
    [90, 40, 30],
    [80, 30, 50]
])

cmyk_values = np.array([
    [0.2, 0.3, 0.4, 0.1],
    [0.25, 0.35, 0.45, 0.15],
    [0.3, 0.4, 0.5, 0.2],
    [0.35, 0.45, 0.55, 0.25],
    [0.4, 0.5, 0.6, 0.3],
    [0.22, 0.32, 0.42, 0.12],
    [0.27, 0.37, 0.47, 0.17],
    [0.32, 0.42, 0.52, 0.22],
    [0.37, 0.47, 0.57, 0.27],
    [0.35, 0.45, 0.50, 0.25],
    [0.38, 0.48, 0.58, 0.28]
])

# Example list of target CIELAB values to convert to CMYK
target_lab_list = np.array([
    [70, 30, 30],
    [60, 20, 15],
    [80, 40, 45]
])

# Convert each target LAB value to corresponding CMYK values
for target_lab in target_lab_list:
    inverse_cmyk_values = inverse_cmyk_characterization(lab_values, cmyk_values, target_lab)
    print(f"Derived CMYK values for target CIELAB {target_lab}: {inverse_cmyk_values}")

