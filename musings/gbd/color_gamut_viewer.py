import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

def plot_3d_color_gamut_with_surface(image_path, output_plot_path):
    # Load the image
    image = io.imread(image_path)
    
    # Convert the image to LAB color space
    lab_image = color.rgb2lab(image)
    
    # Reshape the image to a list of LAB colors
    lab_values = lab_image.reshape((-1, 3))
    
    # Normalize LAB values for color mapping
    L = lab_values[:, 0]
    a = lab_values[:, 1]
    b = lab_values[:, 2]
    
    # Create a convex hull from the Lab values
    hull = ConvexHull(lab_values)
    
    # Convert the Lab values to RGB for the colors
    rgb_colors = color.lab2rgb(lab_values.reshape(-1, 1, 3)).reshape(-1, 3)
    
    # Plot the LAB color gamut as a 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with Lab values
    ax.scatter(a, b, L, c=rgb_colors, s=2, marker='.')
    
    # Plot the surface using the original Lab values and RGB colors
    for simplex in hull.simplices:
        triangle = lab_values[simplex]
        triangle_rgb = color.lab2rgb(triangle.reshape(1, -1, 3)).reshape(-1, 3)
        ax.plot_trisurf(triangle[:, 1], triangle[:, 2], triangle[:, 0],
                        color=triangle_rgb.mean(axis=0), edgecolor='none', alpha=0.9)
    
    # Set axis labels
    ax.set_xlabel('a*')
    ax.set_ylabel('b*')
    ax.set_zlabel('L*')
    
    # Set axis limits
    ax.set_xlim([-128, 127])
    ax.set_ylim([-128, 127])
    ax.set_zlim([0, 100])
    
    # Set the view angle for better visualization
    ax.view_init(elev=20, azim=-30)
    
    # Set title
    ax.set_title('3D Solid Color Gamut in LAB Color Space with RGB-colored Surface')
    
    # Save the plot
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

# Example usage
example_image_path = 'path_to_your_image.jpg'  # Replace with your actual image path
output_plot_path = '3d_lab_gamut_plot_with_surface_rgb.png'
plot_3d_color_gamut_with_surface(example_image_path, output_plot_path)

print(f"3D LAB color gamut plot with RGB-colored surface saved to: {output_plot_path}")
