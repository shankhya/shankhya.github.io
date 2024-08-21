#Plot Gamut of an input image
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.spatial import ConvexHull


def compute_gamut_boundary(image_path, output_path):
    # Load the image
    image = io.imread(image_path)

    # Check if the image has an alpha channel and remove it if necessary
    if image.shape[2] == 4:
        image = image[:, :, :3]  # Remove the alpha channel

    # Convert the image to CIELAB color space
    lab_image = color.rgb2lab(image)

    # Reshape the image to a 2D array where each row is a color in LAB space
    lab_colors = lab_image.reshape((-1, 3))

    # Convert LAB colors back to RGB for plotting
    rgb_colors = color.lab2rgb(lab_colors.reshape(lab_image.shape)).reshape(-1, 3)

    # Compute the convex hull of the LAB colors to get the gamut boundary
    hull = ConvexHull(lab_colors)

    # Plot the gamut boundary in the CIELAB space
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with colors based on LAB values converted to RGB
    ax.scatter(lab_colors[:, 1], lab_colors[:, 2], lab_colors[:, 0], color=rgb_colors, s=1)

    # Set labels and title
    ax.set_xlabel('a*')
    ax.set_ylabel('b*')
    ax.set_zlabel('L*')
    ax.set_title('CIELAB Gamut Boundary')

    # Save the plot to a file
    plt.savefig(output_path)
    plt.close(fig)  # Close the plot to ensure it doesn't display


# Example usage
compute_gamut_boundary('sample.jpg', 'output_plot.png')
