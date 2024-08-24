#single slice
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.spatial import ConvexHull

def plot_gamut_in_lab_space(image_path, output_path='gamut_plot.png'):
    try:
        # Load the image
        image = io.imread(image_path)

        # Convert the image from RGB to LAB color space
        lab_image = color.rgb2lab(image)

        # Reshape the image to a 2D array where each row is a color in LAB space
        lab_pixels = lab_image.reshape(-1, 3)
        rgb_pixels = image.reshape(-1, 3) / 255.0  # Normalize RGB values to [0, 1] range

        # Extract L, a, b channels
        L = lab_pixels[:, 0]
        a = lab_pixels[:, 1]
        b = lab_pixels[:, 2]

        # Perform ConvexHull to find the boundary of the gamut in the LAB space
        hull = ConvexHull(np.c_[a, b])

        # Plot the gamut boundary with RGB colors
        plt.figure(figsize=(8, 8))
        plt.scatter(a, b, color=rgb_pixels, s=1, alpha=0.8)  # Use RGB colors for points
        for simplex in hull.simplices:
            plt.plot(a[simplex], b[simplex], 'k-')

        plt.xlabel('a*')
        plt.ylabel('b*')
        plt.title('Gamut Boundary in LAB Space')
        plt.grid(True)

        # Save the plot instead of showing it
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved as {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
plot_gamut_in_lab_space('sample1.jpg', 'output_gamut_plot.png')
