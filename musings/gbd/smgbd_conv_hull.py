import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage import io, color
# Reference point in LAB space (typically the center of the gamut)
REFERENCE_POINT = np.array([50, 0, 0])
# Pre-conditioning transformation function (based on Bala paper)
def preconditioning_transform(lab_points, reference_point=REFERENCE_POINT, alpha=80, gamma=0.5, eps=1e-6):
    # Calculate the Euclidean distance from each LAB point to the reference point
    D = np.linalg.norm(lab_points - reference_point, axis=1)
    D_max = np.max(D)
    # Clamp D values to avoid division by zero
    D = np.maximum(D, eps)
    # Apply the pre-conditioning transform to the distances
    D_prime = D * ((D_max / np.maximum(D_max - D, eps)) ** gamma)
    # Transform the LAB points by scaling the difference vectors
    direction_vectors = (lab_points - reference_point) / D[:, np.newaxis]  # Unit vectors in direction of LAB points
    transformed_points = reference_point + D_prime[:, np.newaxis] * direction_vectors
    return transformed_points
# Inverse transform to bring points back to original LAB space
def inverse_transform(transformed_points, reference_point=REFERENCE_POINT, alpha=80, gamma=0.5, eps=1e-6):
    # Calculate the Euclidean distance from transformed points to the reference point
    D_prime = np.linalg.norm(transformed_points - reference_point, axis=1)
    D_max = np.max(D_prime)
    # Clamp D_prime values to avoid division by zero
    D_prime = np.maximum(D_prime, eps)
    # Apply inverse of the pre-conditioning transform
    D = D_prime / ((D_max / np.maximum(D_max - D_prime, eps)) ** gamma)
    # Transform back to original LAB space
    direction_vectors = (transformed_points - reference_point) / D_prime[:, np.newaxis]
    original_points = reference_point + D[:, np.newaxis] * direction_vectors
    return original_points
# Filter out NaN values
def filter_invalid_points(points):
    # Remove points that contain NaN or Inf values
    valid_mask = np.isfinite(points).all(axis=1)
    return points[valid_mask]
# Convert LAB to spherical coordinates (r, alpha, theta)
def lab_to_spherical(lab_colors, E=REFERENCE_POINT):
    L = lab_colors[:, 0]
    a = lab_colors[:, 1]
    b = lab_colors[:, 2]
    # Spherical coordinates
    r = np.sqrt((L - E[0]) ** 2 + (a - E[1]) ** 2 + (b - E[2]) ** 2)
    alpha = np.arctan2(b - E[2], a - E[1]) * (180 / np.pi)  # Convert radians to degrees
    alpha = alpha % 360
    theta = np.arctan2(L - E[0], np.sqrt((a - E[1]) ** 2 + (b - E[2]) ** 2)) * (180 / np.pi)  # Convert to degrees
    return r, alpha, theta
# Segment maxima method to find the extrema points
def segment_maxima(lab_pixels, n_alpha=1000, n_theta=500, alpha=80, gamma=0.5):
    # Apply preconditioning transform to the LAB points
    transformed_lab = preconditioning_transform(lab_pixels, alpha=alpha, gamma=gamma)
    r, alpha_angles, theta = lab_to_spherical(transformed_lab)
    # Create finer bins for segmentation
    alpha_bins = np.linspace(0, 360, n_alpha)
    theta_bins = np.linspace(-90, 90, n_theta)
    maxima = []
    for i in range(n_alpha - 1):
        for j in range(n_theta - 1):
            # Mask points that fall into the current segment
            mask = (alpha_angles >= alpha_bins[i]) & (alpha_angles < alpha_bins[i + 1]) & \
                   (theta >= theta_bins[j]) & (theta < theta_bins[j + 1])
            segment_points = transformed_lab[mask]
            if len(segment_points) > 0:
                # Find the point with the maximum radius
                r_segment = r[mask]
                max_idx = np.argmax(r_segment)
                maxima.append(segment_points[max_idx])
    maxima = np.array(maxima)
    # Apply inverse transform to bring points back to original LAB space
    maxima_points = inverse_transform(maxima, alpha=alpha, gamma=gamma)
    # Filter out any invalid points (NaN, Inf)
    maxima_points = filter_invalid_points(maxima_points)
    return maxima_points
# Create a convex hull
def create_convex_hull(points):
    if len(points) < 4:
        return None
    hull = ConvexHull(points)
    return hull
# Plot the GBD with colors based on LAB values
def plot_gamut_boundary(hull, points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Convert LAB points to RGB for coloring the segments
    lab_colors = points
    rgb_colors = color.lab2rgb(lab_colors[np.newaxis, :, :]).reshape(-1, 3)
    # Plot the convex hull triangles with color mapping
    for simplex in hull.simplices:
        triangle = points[simplex]
        rgb_triangle = rgb_colors[simplex]
        avg_rgb = np.mean(rgb_triangle, axis=0)  # Average RGB color for the face
        ax.plot_trisurf(triangle[:, 1], triangle[:, 2], triangle[:, 0], color=avg_rgb, shade=True)
    # Set axis labels
    ax.set_xlabel('a* (Green-Red)')
    ax.set_ylabel('b* (Blue-Yellow)')
    ax.set_zlabel('L* (Lightness)')
    plt.show()
# Load image and convert to LAB space
image = io.imread('lena.png')
lab_image = color.rgb2lab(image)
lab_pixels = lab_image.reshape(-1, 3)
# Compute segment maxima points using modified convex hull method with more precise segmentation
maxima_points = segment_maxima(lab_pixels, n_alpha=1000, n_theta=500)  # Increased granularity for smaller triangles
# Create convex hull from maxima points
hull = create_convex_hull(maxima_points)
# Plot the GBD
if hull is not None:
    plot_gamut_boundary(hull, maxima_points)
else:
    print("Not enough points for a valid convex hull")