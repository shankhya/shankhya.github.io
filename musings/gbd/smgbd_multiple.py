#SMGBD multiple views
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage import color
from PIL import Image
def create_gbd_surface_multiple_views(image_path, output_path):
    """
    Creates multiple views of a GBD gamut surface from an input image and saves them in a single image.
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
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': '3d'})
    # Define the view angles (elevation and azimuth)
    views = [(20, -135), (30, 45), (60, -45), (90, 90)]
    # Loop over the subplots and set different views
    for ax, (elev, azim) in zip(axs.flat, views):
        for simplex in hull.simplices:
            triangle = lab_data[simplex]
            rgb_color = rgb_data[simplex]
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], color=rgb_color.mean(axis=0), edgecolor='none', alpha=1)
        ax.set_xlabel('a*')
        ax.set_ylabel('b*')
        ax.set_zlabel('L*')
        ax.view_init(elev=elev, azim=azim)
    # Save all plots in a single image
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
# Example usage
image_path = "sample.jpg"
output_path = "gbd_surface_multiple_views.png"
create_gbd_surface_multiple_views(image_path, output_path)
