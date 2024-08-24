#RBF based GBD
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, color
from scipy.interpolate import Rbf

def plot_gamut_rbf(image_path, output_path='gamut_rbf_output.png'):
    try:
        # Load the image
        image = io.imread(image_path)

        # Downsample the image for faster processing and reduced memory usage
        downsample_factor = 10  # Increase this factor to reduce memory further
        image = image[::downsample_factor, ::downsample_factor]

        # Convert the image from RGB to LAB color space
        lab_image = color.rgb2lab(image)

        # Reshape the image to a 2D array where each row is a color in LAB space
        lab_pixels = lab_image.reshape(-1, 3)
        L, a, b = lab_pixels[:, 0], lab_pixels[:, 1], lab_pixels[:, 2]

        # Reduce the number of grid points
        grid_size = 50  # Decrease this number to reduce memory usage
        a_grid, b_grid = np.meshgrid(np.linspace(a.min(), a.max(), grid_size),
                                     np.linspace(b.min(), b.max(), grid_size))

        # Fit RBF (Radial Basis Function) to approximate the gamut boundary
        rbf = Rbf(a, b, L, function='multiquadric', smooth=0.1)

        # Predict the L* values for the grid using the RBF
        L_grid = rbf(a_grid, b_grid)

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the original points
        ax.scatter(a, b, L, color=np.clip(lab_pixels / 100.0, 0, 1), s=1, alpha=0.3)

        # Plot the RBF surface
        ax.plot_surface(a_grid, b_grid, L_grid, color='cyan', alpha=0.3, edgecolor='none')

        ax.set_xlabel('a*')
        ax.set_ylabel('b*')
        ax.set_zlabel('L*')
        ax.set_title('Gamut Boundary Approximation using RBF in L*a*b* Space')

        # Save the plot without showing it
        plt.savefig(output_path)
        plt.close()
        print(f"3D RBF plot saved as {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
plot_gamut_rbf('sample1.jpg', 'gamut_rbf_output_optimized.png')
