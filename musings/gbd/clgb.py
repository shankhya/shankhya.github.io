#CLGB multiple angles
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from skimage import color
from PIL import Image
def lab_to_lch(lab):
    """
    Convert LAB to LCH (Lightness, Chroma, Hue).
    """
    L, A, B = lab[:, 0], lab[:, 1], lab[:, 2]
    C = np.sqrt(A**2 + B**2)
    H = np.degrees(np.arctan2(B, A))
    H[H < 0] += 360  # Ensure H is in [0, 360]
    return np.stack((L, C, H), axis=-1)
def clgb_gbd_intersections(image_path, output_path, target_hues, hue_tolerance=5.0):
    """
    Implements the CLGB Method for GBD plotting intersected by planes of constant hue angles.
    Args:
        image_path: Path to the input image file.
        output_path: Path to save the output image.
        target_hues: List of hue angles to use for the intersections.
        hue_tolerance: Tolerance for matching hue angles (default is 5.0 degrees).
    """
    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)
    # Convert RGB to LAB color space
    lab_array = color.rgb2lab(img_array / 255.0)
    # Reshape LAB values to a 2D array
    L, A, B = lab_array[:, :, 0], lab_array[:, :, 1], lab_array[:, :, 2]
    lab_data = np.column_stack((L.flatten(), A.flatten(), B.flatten()))
    # Convert LAB to LCH to get hue angles
    lch_data = lab_to_lch(lab_data)
    # Set up the figure for multiple subplots
    num_hues = len(target_hues)
    fig, axs = plt.subplots(1, num_hues, figsize=(5 * num_hues, 5), sharey=True)
    if num_hues == 1:
        axs = [axs]  # Ensure axs is iterable if only one subplot
    # Loop through each target hue and plot the corresponding GBD intersection
    for ax, target_hue in zip(axs, target_hues):
        # Find points within a small range of the target hue
        close_hue_indices = np.where(np.abs(lch_data[:, 2] - target_hue) < hue_tolerance)[0]
        close_hue_points = lab_data[close_hue_indices]
        # If not enough points are found, increase tolerance or skip the hue
        if len(close_hue_points) < 3:
            if hue_tolerance < 10:
                print(f"Not enough points found for hue {target_hue} with tolerance {hue_tolerance}. Increasing tolerance.")
                close_hue_indices = np.where(np.abs(lch_data[:, 2] - target_hue) < hue_tolerance * 2)[0]
                close_hue_points = lab_data[close_hue_indices]
            if len(close_hue_points) < 3:
                print(f"Skipping hue {target_hue} due to insufficient points.")
                ax.set_visible(False)
                continue
        # Perform Delaunay triangulation on the points near the target hue
        tri = Delaunay(close_hue_points[:, [1, 2]])  # Triangulate on a* and b* (ignoring L*)
        # Convert the selected LAB points back to RGB for coloring the 2D boundary
        close_hue_lab = close_hue_points
        rgb_data = color.lab2rgb(close_hue_lab.reshape(-1, 1, 3)).reshape(-1, 3)
        # Plot the 2D GBD boundary formed by the intersection
        for simplex in tri.simplices:
            triangle = close_hue_lab[simplex]
            rgb_color = rgb_data[simplex]
            ax.fill(triangle[:, 1], triangle[:, 2], color=rgb_color.mean(axis=0), edgecolor='k', alpha=0.8)
        ax.set_xlabel('a*')
        ax.set_ylabel('b*')
        ax.set_title(f'Hue: {target_hue}°')
    # Adjust layout and save the plot as an image
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # Close the figure
    plt.close()
# Example usage
image_path = "sample.jpg"
output_path = "clgb_gbd_hue_intersections.png"
# Prompt user for hue angles
target_hues_input = input("Enter the hue angles separated by commas (e.g., 30, 120, 240): ")
target_hues = [float(hue.strip()) for hue in target_hues_input.split(",")]
clgb_gbd_intersections(image_path, output_path, target_hues)
