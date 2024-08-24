#slices
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.spatial import ConvexHull

def plot_gamut_slices(image_path, output_path='gamut_slices_plot.png', L_values=None):
    try:
        # Load the image
        image = io.imread(image_path)

        # Convert the image from RGB to LAB color space
        lab_image = color.rgb2lab(image)

        # Reshape the image to a 2D array where each row is a color in LAB space
        lab_pixels = lab_image.reshape(-1, 3)
        rgb_pixels = image.reshape(-1, 3) / 255.0  # Normalize RGB values to [0, 1] range

        # If L_values is not provided, ask the user for input
        if L_values is None:
            L_input = input("Enter the L* values you want to plot (comma-separated, e.g., 20,40,60,80): ")
            L_values = [float(L.strip()) for L in L_input.split(',')]

        # Create subplots for each L* value
        num_slices = len(L_values)
        fig, axes = plt.subplots(1, num_slices, figsize=(16, 8))

        for i, L in enumerate(L_values):
            # Filter the points close to the desired L* value
            tolerance = 2.5  # tolerance for L* value to include in the slice
            mask = np.abs(lab_pixels[:, 0] - L) < tolerance
            a_slice = lab_pixels[mask, 1]
            b_slice = lab_pixels[mask, 2]
            rgb_slice = rgb_pixels[mask]

            # Plot the points in a*-b* space
            ax = axes[i] if num_slices > 1 else axes
            ax.scatter(a_slice, b_slice, color=rgb_slice, s=1, alpha=0.8)
            ax.set_title(f'L* ≈ {L}')
            ax.set_xlabel('a*')
            ax.set_ylabel('b*')
            ax.grid(True)

            # Optionally, plot the convex hull
            if len(a_slice) > 2:  # Convex hull requires at least 3 points
                hull = ConvexHull(np.c_[a_slice, b_slice])
                for simplex in hull.simplices:
                    ax.plot(a_slice[simplex], b_slice[simplex], 'k-')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved as {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
plot_gamut_slices('sample1.jpg', 'gamut_slices_output.png')
