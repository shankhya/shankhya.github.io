import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage import io, color
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Read LAB values from an Excel file
lab_df = pd.read_excel('lab_data_plot.xlsx')  # Replace 'lab_data_plot.xlsx' with your Excel file name
L_values = lab_df['L*']
a_values = lab_df['a*']
b_values = lab_df['b*']

# Convert all LAB colors to RGB colors and format them
rgb_colors = []
for L, a, b in zip(L_values, a_values, b_values):
    lab_color = LabColor(L, a, b)
    rgb_color = convert_color(lab_color, sRGBColor)
    rgb_color = [rgb_color.rgb_r * 255, rgb_color.rgb_g * 255, rgb_color.rgb_b * 255]
    rgb_colors.append([rgb_color[0] / 255, rgb_color[1] / 255, rgb_color[2] / 255])

# Create a 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Combine LAB values into one array to create a convex hull
lab_points = np.column_stack((a_values, b_values, L_values))

# Generate the convex hull for the LAB points (outer surface only)
hull = ConvexHull(lab_points)

# Plot the GBD (Convex Hull) by plotting the outer triangles formed by the hull's simplices
for simplex in hull.simplices:
    triangle = lab_points[simplex]
    # Get the RGB colors for the triangle vertices
    triangle_rgb = [rgb_colors[simplex[0]], rgb_colors[simplex[1]], rgb_colors[simplex[2]]]
    avg_rgb = np.mean(triangle_rgb, axis=0)  # Average color of the triangle
    # Create a polygon for each triangle in the convex hull
    poly = Poly3DCollection([triangle], facecolor=avg_rgb, edgecolor='none', alpha=0.9)
    ax.add_collection3d(poly)

# Axis lines for reference (optional)
ax.plot([0, 128], [0, 0], [50, 50], color='black', linewidth=2)
ax.plot([-128, 0], [0, 0], [50, 50], color='black', linewidth=2)
ax.plot([0, 0], [-128, 0], [50, 50], color='black', linewidth=2)
ax.plot([0, 0], [0, 128], [50, 50], color='black', linewidth=2)
ax.plot([0, 0], [0, 0], [0, 100], color='black', linewidth=2)

# Set axis labels
ax.set_xlabel('a* (Green-Red)')
ax.set_ylabel('b* (Blue-Yellow)')
ax.set_zlabel('L* (Lightness)')

# Set axis limits
ax.set_xlim([-128, 128])
ax.set_ylim([-128, 128])
ax.set_zlim([0, 100])

# Set plot title
ax.set_title('GBD for the FOGRA39 dataset')

plt.show()
