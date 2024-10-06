#Forward characterization
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Function to convert CMYK to RGB with clamping
def cmyk_to_rgb(c, m, y, k):
    r = 1 - min(1, c * (1 - k) + k)
    g = 1 - min(1, m * (1 - k) + k)
    b = 1 - min(1, y * (1 - k) + k)
    return np.clip([r, g, b], 0, 1)  # Clamp to 0-1 range

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

# Define ranges for interpolation
L_range = np.linspace(0, 100, 20)  # L* range from 0 to 100
a_range = np.linspace(-128, 128, 20)  # a* range from -128 to 128
b_range = np.linspace(-128, 128, 20)  # b* range from -128 to 128

# Create a meshgrid of L*, a*, and b* values
L_grid, a_grid, b_grid = np.meshgrid(L_range, a_range, b_range)
grid_points = np.vstack([L_grid.ravel(), a_grid.ravel(), b_grid.ravel()]).T

# Extract L* values for interpolation
L_values = lab_values[:, 0]

# Initialize arrays for interpolated CMYK values
C_interp = np.zeros(grid_points.shape[0])
M_interp = np.zeros(grid_points.shape[0])
Y_interp = np.zeros(grid_points.shape[0])
K_interp = np.zeros(grid_points.shape[0])

# Perform piecewise linear interpolation for each CMYK channel
for i in range(4):  # For each CMYK channel
    interp_func = interp1d(L_values, cmyk_values[:, i], kind='linear', fill_value='extrapolate')
    if i == 0:  # Cyan
        C_interp = interp_func(grid_points[:, 0])
    elif i == 1:  # Magenta
        M_interp = interp_func(grid_points[:, 0])
    elif i == 2:  # Yellow
        Y_interp = interp_func(grid_points[:, 0])
    elif i == 3:  # Black
        K_interp = interp_func(grid_points[:, 0])

# Reshape the interpolated CMYK values to match the grid shape
C_grid = C_interp.reshape(L_grid.shape)
M_grid = M_interp.reshape(L_grid.shape)
Y_grid = Y_interp.reshape(L_grid.shape)
K_grid = K_interp.reshape(L_grid.shape)

# Create a 3D plot of the CMYK channels in subplots
fig = plt.figure(figsize=(15, 12))

# Plot each CMYK channel in a separate subplot
for i, (color_data, color_name, cmap) in enumerate(zip(
        [C_grid, M_grid, Y_grid, K_grid],
        ['Cyan', 'Magenta', 'Yellow', 'Black'],
        ['Blues', 'Reds', 'YlGn', 'Greys'])):
    
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    
    # Plot using the CMYK colors for each point
    sc = ax.scatter(a_grid, b_grid, L_grid, c=color_data, cmap=cmap, marker='o', alpha=0.5)

    # Add a color bar for visualization
    plt.colorbar(sc, ax=ax, label=f'{color_name} Ink Coverage')

    # Label the axes
    ax.set_xlabel('a* (Green-Red)')
    ax.set_ylabel('b* (Blue-Yellow)')
    ax.set_zlabel('L* (Lightness)')
    ax.set_title(f'Interpolated {color_name} Values in the CIELAB Space')

    # Invert L* axis for visualization
    ax.set_zlim(100, 0)

plt.tight_layout()
plt.show()
