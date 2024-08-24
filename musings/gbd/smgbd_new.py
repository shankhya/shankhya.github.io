#SMGBD
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage import color
from PIL import Image
def create_gbd_surface(image_path, output_path):
    """
    Creates a GBD gamut surface from an input image and saves it as an image.
    Args:
        image_path: Path to the input image file.
        output_path: Path to save the output image.
    """
    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)
    # Convert RGB to LAB color space
    lab_array = color.rgb2lab(img_array / 255.0)  # Normalize to [0, 1] for skimage
    # Extract LAB values and reshape to a 2D array
    L, A, B = lab_array[:, :, 0], lab_array[:, :, 1], lab_array[:, :, 2]
    lab_data = np.column_stack((A.flatten(), B.flatten(), L.flatten()))  # Reorder to make L* vertical (z-axis)
    # Calculate the convex hull of the LAB data
    hull = ConvexHull(lab_data)
    # Convert LAB data back to RGB
    lab_data_original_order = np.column_stack((L.flatten(), A.flatten(), B.flatten()))  # Use original LAB order for RGB conversion
    rgb_data = color.lab2rgb(lab_data_original_order.reshape(-1, 1, 3)).reshape(-1, 3)
    # Create the GBD gamut surface plot using the reordered lab_data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the triangulated surface, coloring according to the RGB values
    for simplex in hull.simplices:
        triangle = lab_data[simplex]
        rgb_color = rgb_data[simplex]
        ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], color=rgb_color.mean(axis=0), edgecolor='none', alpha=1)
    ax.set_xlabel('a*')
    ax.set_ylabel('b*')
    ax.set_zlabel('L*')  # L* is now the vertical axis
    # Improve the visual appearance with better lighting and angles
    ax.view_init(elev=20, azim=-135)
    # ax.dist = 7  # This line is removed due to deprecation
    # Save the plot as an image
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # Close the figure (optional)
    plt.close()
# Example usage
image_path = "sample.jpg"
output_path = "gbd_surface.png"
create_gbd_surface(image_path, output_path)