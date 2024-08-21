#SMGBD & convex hull
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def calculate_gbd_segment_maxima(image, m=16, n=16):
    # Convert the image to LAB color space
    lab_image = color.rgb2lab(image)
    lab_flat = lab_image.reshape(-1, 3)

    # Convert LAB to spherical coordinates (using the formula for spherical coordinates in LAB)
    L, a, b = lab_flat[:, 0], lab_flat[:, 1], lab_flat[:, 2]
    radius = np.sqrt(a ** 2 + b ** 2)
    theta = np.degrees(np.arctan2(b, a)) % 360
    phi = np.degrees(np.arccos(L / 100.0))

    # Initialize the GBD matrix
    gbd_matrix = np.full((m, n, 3), -np.inf)

    # Populate the GBD matrix
    for i in range(len(L)):
        aindex = min(int(np.floor(theta[i] / (360 / m))), m - 1)
        yindex = min(int(np.floor(phi[i] / (180 / n))), n - 1)

        if radius[i] > gbd_matrix[aindex, yindex, 0]:
            gbd_matrix[aindex, yindex] = [radius[i], theta[i], phi[i]]

    return gbd_matrix


def plot_gbd_with_convex_hull(gbd_matrix):
    # Convert spherical coordinates back to LAB for plotting
    radii = gbd_matrix[:, :, 0].flatten()
    thetas = np.radians(gbd_matrix[:, :, 1].flatten())
    phis = np.radians(gbd_matrix[:, :, 2].flatten())

    a_values = radii * np.cos(thetas)
    b_values = radii * np.sin(thetas)
    L_values = 100 * np.cos(phis)

    # Stack the LAB values to form the point cloud
    points = np.stack([L_values, a_values, b_values], axis=-1)

    # Filter out invalid points
    valid_points = points[np.isfinite(points).all(axis=1)]

    # Calculate the convex hull
    hull = ConvexHull(valid_points)

    # Plot the points and the convex hull
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the LAB points
    ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
               c=color.lab2rgb(valid_points.reshape(-1, 1, 3)).reshape(-1, 3), alpha=0.6)

    # Plot the convex hull
    for simplex in hull.simplices:
        ax.add_collection3d(Poly3DCollection([valid_points[simplex]], alpha=0.3, facecolor='cyan'))

    ax.set_xlabel('L*')
    ax.set_ylabel('a*')
    ax.set_zlabel('b*')
    ax.set_title('Gamut Boundary Descriptor with Convex Hull in LAB Space')

    # Save the plot without displaying it
    plt.savefig('gamut_boundary_descriptor_convex_hull_lab_space.png')
    plt.close()


# Load your image
image = io.imread('sample.jpg')  # replace with your image path

# Calculate the GBD
gbd_matrix = calculate_gbd_segment_maxima(image, m=16, n=16)

# Plot and save the GBD with the Convex Hull
plot_gbd_with_convex_hull(gbd_matrix)

