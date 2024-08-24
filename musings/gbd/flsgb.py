#FLSGB
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io


def convert_to_lab(image):
    """Convert an image from RGB to CIELAB color space."""
    return color.rgb2lab(image)


def calculate_gbd(image_lab, num_segments=16):
    """Calculate the Gamut Boundary Descriptor (GBD) using the Segment Maxima method."""
    L, a, b = image_lab[:, :, 0], image_lab[:, :, 1], image_lab[:, :, 2]
    r = np.sqrt(L ** 2 + a ** 2 + b ** 2)
    u = np.arctan2(b, a)  # Angle in the L*a*b* plane
    a_spherical = np.arctan2(L, np.sqrt(a ** 2 + b ** 2))

    a_bins = np.linspace(np.min(a_spherical), np.max(a_spherical), num_segments + 1)
    u_bins = np.linspace(np.min(u), np.max(u), num_segments + 1)

    gbd = np.zeros((num_segments, num_segments, 3))

    for i in range(num_segments):
        for j in range(num_segments):
            mask = (a_spherical >= a_bins[i]) & (a_spherical < a_bins[i + 1]) & \
                   (u >= u_bins[j]) & (u < u_bins[j + 1])
            if np.any(mask):
                index = np.argmax(r[mask])
                gbd[i, j] = [L[mask][index], a[mask][index], b[mask][index]]

    return gbd, a_bins, u_bins


def plot_gamut_boundary_and_intersections(image_lab, gbd, a_bins, u_bins, target_hue):
    """Plot the gamut boundary intersection at the target hue."""
    hue_idx = np.argmin(np.abs(u_bins - target_hue))

    # Extract the L*, a*, b* values at the target hue
    target_gamut = gbd[:, hue_idx, :]

    # Filter out empty entries in the GBD
    target_gamut = target_gamut[np.any(target_gamut != 0, axis=1)]

    # Calculate the boundary polygon (2D in this case)
    L_values = target_gamut[:, 0]
    a_values = np.sqrt(target_gamut[:, 1] ** 2 + target_gamut[:, 2] ** 2)  # Chroma

    plt.figure(figsize=(8, 6))

    # Plot the gamut boundary
    plt.plot(a_values, L_values, marker='o', linestyle='-', color='b', label='Gamut Boundary')

    # Create lines (l) and calculate intersections
    for slope in [0.5, 1.0, 1.5]:  # Example slopes
        l_a = np.linspace(np.min(a_values), np.max(a_values), 100)
        l_L = slope * l_a + 20  # Example line equation L* = slope * a + intercept

        plt.plot(l_a, l_L, linestyle='--', label=f'Line with slope {slope}')

        # Find intersection points with the gamut boundary
        for i in range(len(a_values) - 1):
            if (l_L[i] > L_values[i] and l_L[i] < L_values[i + 1]) or (
                    l_L[i] < L_values[i] and l_L[i] > L_values[i + 1]):
                intersection_a = l_a[i]
                intersection_L = l_L[i]
                plt.plot(intersection_a, intersection_L, 'rx')  # Mark intersection point

    plt.title(f'Gamut Boundary Intersection at Hue {target_hue:.2f} radians')
    plt.xlabel('Chroma (C*)')
    plt.ylabel('Lightness (L*)')
    plt.grid(True)
    plt.legend()
    plt.savefig('gamut_boundary_intersection.png')
    print("Plot saved as 'gamut_boundary_intersection.png'")


def apply_fslgb_and_plot(image_path, target_hue, num_segments=16):
    """Main function to apply the FSLGB method and plot the intersection."""
    image = io.imread(image_path)
    image_lab = convert_to_lab(image)

    gbd, a_bins, u_bins = calculate_gbd(image_lab, num_segments=num_segments)
    plot_gamut_boundary_and_intersections(image_lab, gbd, a_bins, u_bins, target_hue)


# Example usage
image_path = 'sample.jpg'  # Replace with your image file path
target_hue = np.pi / 4  # Example hue angle in radians (45 degrees)
apply_fslgb_and_plot(image_path, target_hue)
