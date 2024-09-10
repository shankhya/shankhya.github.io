import cv2
import pandas as pd
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from skimage.color import deltaE_ciede2000  # Import for Delta E 2000 calculation


# Function to convert LAB to LCH and back
def lab_to_lch(L, a, b):
    C = np.sqrt(a ** 2 + b ** 2)  # Chroma (C)
    H = np.arctan2(b, a)  # Hue angle (in radians)
    H = np.degrees(H)  # Convert hue to degrees
    H[H < 0] += 360  # Ensure hue is between 0 and 360 degrees
    return L, C, H


def lch_to_lab(L, C, H):
    H_rad = np.radians(H)  # Convert hue back to radians
    a = C * np.cos(H_rad)  # Compute a from C and H
    b = C * np.sin(H_rad)  # Compute b from C and H
    return L, a, b


# Load the image and convert to LAB, then LCH
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = color.rgb2lab(image_rgb)
    return image_rgb, image_lab


# Extract LAB to LCH and Convex Hull
def get_lch_and_hull(image_lab):
    # Extract L, a, b from the image
    L_image = image_lab[..., 0].flatten()
    a_image = image_lab[..., 1].flatten()
    b_image = image_lab[..., 2].flatten()

    # Convert image LAB to LCH
    L_image, C_image, H_image = lab_to_lch(L_image, a_image, b_image)

    # Create convex hull for image data
    points_image = np.column_stack((C_image, L_image))  # Combine C and L values for image
    hull_image = ConvexHull(points_image)

    return L_image, C_image, H_image, hull_image


# Function to check if a point is inside the CMYK convex hull
def is_point_in_hull(point, hull):
    """ Check if a point is inside the given convex hull """
    return np.all(np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1] <= 0)


# Function to find the nearest point on the hull boundary
def map_point_to_hull_boundary(point, hull_points):
    """ Map a point outside the hull to the nearest point on the hull boundary """
    distances = cdist([point], hull_points, metric='euclidean')
    nearest_index = np.argmin(distances)
    nearest_point = hull_points[nearest_index]
    return nearest_point


# Perform the compression and mapping to CMYK boundary and towards L=50
def compress_to_cmyk_boundary(L_image, C_image, hull_cmyk, points_cmyk, L_target=50, factor=0.5):
    compressed_L, compressed_C = [], []

    for L, C in zip(L_image, C_image):
        point = np.array([C, L])

        nearest_point = map_point_to_hull_boundary(point, points_cmyk)
        nearest_L, nearest_C = nearest_point[1], nearest_point[0]

        # Always move towards L_target = 50
        direction_L = L_target - L

        # Check if point is inside the CMYK hull
        if is_point_in_hull(point, hull_cmyk):
            # In-gamut colors: compress towards L=50 and CMYK boundary proportionally
            compressed_L.append(L + factor * (nearest_L - L + direction_L))  # Proportional movement towards L=50
            compressed_C.append(
                min(C + factor * (nearest_C - C), nearest_C))  # Ensure chroma doesn't exceed boundary chroma
        else:
            # Out-of-gamut colors: move directly to the boundary and compress L towards 50
            compressed_L.append(nearest_L + 0.4 * direction_L)  # Move directly to the boundary
            compressed_C.append(nearest_C)

    # After compression, ensure no point is outside the CMYK boundary
    compressed_L, compressed_C = np.array(compressed_L), np.array(compressed_C)
    for i in range(len(compressed_L)):
        point = np.array([compressed_C[i], compressed_L[i]])
        if not is_point_in_hull(point, hull_cmyk):
            # If still outside the CMYK gamut, map it directly to the CMYK boundary
            nearest_point = map_point_to_hull_boundary(point, points_cmyk)
            compressed_L[i], compressed_C[i] = nearest_point[1], nearest_point[0]

    return compressed_L, compressed_C


# Plot Gamut Convex Hulls for Original, CMYK, and Compressed Image
def plot_gamut_convex_hulls(L_image, C_image, compressed_L, compressed_C, hull_image, hull_cmyk, points_cmyk):
    plt.figure(figsize=(10, 8))

    # Plot original RGB convex hull (solid blue line)
    image_hull_points = np.append(hull_image.vertices, hull_image.vertices[0])
    points_image = np.column_stack((C_image, L_image))
    plt.plot(points_image[image_hull_points, 0], points_image[image_hull_points, 1], 'b-', linewidth=2,
             label='Original image gamut')

    # Plot CMYK convex hull (dashed green line)
    cmyk_hull_points = np.append(hull_cmyk.vertices, hull_cmyk.vertices[0])
    plt.plot(points_cmyk[cmyk_hull_points, 0], points_cmyk[cmyk_hull_points, 1], 'g--', linewidth=2, label='CMYK Gamut')

    # Plot compressed RGB convex hull (solid red line)
    points_compressed = np.column_stack((compressed_C, compressed_L))
    hull_compressed = ConvexHull(points_compressed)
    compressed_hull_points = np.append(hull_compressed.vertices, hull_compressed.vertices[0])
    plt.plot(points_compressed[compressed_hull_points, 0], points_compressed[compressed_hull_points, 1], 'r-',
             linewidth=2, label='Compressed image gamut')

    plt.xlabel("Chroma (C)")
    plt.ylabel("Lightness (L)")
    plt.title("Comparison of gamuts")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


# Display side-by-side comparison of the original and compressed images
def display_images(original_rgb, compressed_rgb_image):
    plt.figure(figsize=(16, 8))  # Make the figure wider for side-by-side display

    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # Display compressed image
    plt.subplot(1, 2, 2)
    plt.imshow(compressed_rgb_image)
    plt.title('Gamut Compression towards L=50')
    plt.axis('off')

    # Show the combined plot
    plt.show()


# Calculate Delta E 2000 between the original and compressed images
def calculate_delta_e(original_lab, compressed_lab):
    # Flatten the LAB images for calculation
    original_lab_flat = original_lab.reshape((-1, 3))
    compressed_lab_flat = compressed_lab.reshape((-1, 3))

    # Compute Delta E 2000
    delta_e_2000 = deltaE_ciede2000(original_lab_flat, compressed_lab_flat)

    # Print the overall average Delta E 2000 value
    avg_delta_e = np.mean(delta_e_2000)
    print(f"Average Delta E 2000 between original and compressed image: {avg_delta_e:.4f}")


# Main function to load images, perform compression, and display results
def main():
    # Load RGB image
    image_path = 'newdataset/sample.jpg'
    image_rgb, image_lab = load_image(image_path)

    # Get LCH values and convex hull for RGB image
    L_image, C_image, H_image, hull_image = get_lch_and_hull(image_lab)

    # Load Fogra39 CMYK LAB data
    csv_path = 'newdataset/lab.csv'
    data = pd.read_csv(csv_path)

    # Extract L, a, b from the CSV file
    L_csv = data['L'].values
    a_csv = data['a'].values
    b_csv = data['b'].values

    # Convert CSV LAB to LCH and get convex hull for CMYK
    L_csv, C_csv, H_csv = lab_to_lch(L_csv, a_csv, b_csv)
    points_cmyk = np.column_stack((C_csv, L_csv))
    hull_cmyk = ConvexHull(points_cmyk)

    # Perform compression and mapping
    compressed_L, compressed_C = compress_to_cmyk_boundary(L_image, C_image, hull_cmyk, points_cmyk)

    # Visualize the convex hulls
    plot_gamut_convex_hulls(L_image, C_image, compressed_L, compressed_C, hull_image, hull_cmyk, points_cmyk)

    # Convert compressed LCH back to LAB, and then to RGB
    compressed_Lab_L, compressed_Lab_a, compressed_Lab_b = lch_to_lab(compressed_L, compressed_C, H_image)
    compressed_Lab_L = compressed_Lab_L.reshape(image_lab.shape[:2])
    compressed_Lab_a = compressed_Lab_a.reshape(image_lab.shape[:2])
    compressed_Lab_b = compressed_Lab_b.reshape(image_lab.shape[:2])

    compressed_lab_image = np.stack([compressed_Lab_L, compressed_Lab_a, compressed_Lab_b], axis=2)
    compressed_rgb_image = color.lab2rgb(compressed_lab_image)

    # Display original and compressed images side by side
    display_images(image_rgb, compressed_rgb_image)

    # Calculate Delta E 2000 between the original and compressed image
    calculate_delta_e(image_lab, compressed_lab_image)


# Run the main function
main()