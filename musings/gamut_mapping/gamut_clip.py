import cv2
import pandas as pd
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Function to convert LAB to LCH and back
def lab_to_lch(L, a, b):
    C = np.sqrt(a**2 + b**2)  # Chroma (C)
    H = np.arctan2(b, a)      # Hue angle (in radians)
    H = np.degrees(H)         # Convert hue to degrees
    H[H < 0] += 360           # Ensure hue is between 0 and 360 degrees
    return L, C, H

def lch_to_lab(L, C, H):
    H_rad = np.radians(H)  # Convert hue back to radians
    a = C * np.cos(H_rad)  # Compute a from C and H
    b = C * np.sin(H_rad)  # Compute b from C and H
    return L, a, b

# Part 1: Plotting the RGB and CMYK (Fogra39) gamuts
def plot_gamuts():
    # Load the image and convert to LAB, then LCH
    image_path = 'newdataset/sample.jpg'  # Update with your image path
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = color.rgb2lab(image_rgb)

    # Extract L, a, b from the image
    L_image = image_lab[..., 0].flatten()
    a_image = image_lab[..., 1].flatten()
    b_image = image_lab[..., 2].flatten()

    # Convert image LAB to LCH
    L_image, C_image, H_image = lab_to_lch(L_image, a_image, b_image)

    # Load the Fogra39 data from CSV and convert to LCH
    csv_path = 'newdataset/lab.csv'  # Update with your CSV path
    data = pd.read_csv(csv_path)

    # Extract L, a, b from the CSV file
    L_csv = data['L'].values
    a_csv = data['a'].values
    b_csv = data['b'].values

    # Convert CSV LAB to LCH
    L_csv, C_csv, H_csv = lab_to_lch(L_csv, a_csv, b_csv)

    # Create convex hulls for both image data and CSV data
    points_image = np.column_stack((C_image, L_image))  # Combine C and L values for image
    points_csv = np.column_stack((C_csv, L_csv))        # Combine C and L values for CSV

    hull_image = ConvexHull(points_image)
    hull_csv = ConvexHull(points_csv)

    # Plot the combined convex hulls
    plt.figure(figsize=(8, 6))

    # Plot the convex hull for image data (solid black lines) with the label 'RGB gamut'
    image_hull_points = np.append(hull_image.vertices, hull_image.vertices[0])
    plt.plot(points_image[image_hull_points, 0], points_image[image_hull_points, 1], 'k-', linewidth=2, label='Image gamut')

    # Plot the convex hull for Fogra39 data (dashed black lines) with the label 'CMYK gamut'
    csv_hull_points = np.append(hull_csv.vertices, hull_csv.vertices[0])
    plt.plot(points_csv[csv_hull_points, 0], points_csv[csv_hull_points, 1], 'k--', linewidth=2, label='Fogra39 gamut')

    # Set X-axis to start from 0
    plt.xlim(0, max(np.max(C_image), np.max(C_csv)))  # Ensure both datasets fit in the plot

    # Add labels, title, and legend
    plt.xlabel('Chroma (C)')
    plt.ylabel('Lightness (L)')
    plt.title('Gamuts for the test image and the Fogra39 dataset')

    # Add legend to the top-right corner
    plt.legend(loc='upper right')
    plt.grid(True)

    # Show plot
    plt.show()

# Part 2: Gamut mapping
def gamut_mapping():
    # Load the image and convert to LAB, then LCH
    image_path = 'newdataset/sample.jpg'  # Update with your image path
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = color.rgb2lab(image_rgb)

    # Extract L, a, b from the image
    L_image = image_lab[..., 0].flatten()
    a_image = image_lab[..., 1].flatten()
    b_image = image_lab[..., 2].flatten()

    # Convert image LAB to LCH
    L_image, C_image, H_image = lab_to_lch(L_image, a_image, b_image)

    # Load the Fogra39 data from CSV and convert to LCH
    csv_path = 'newdataset/lab.csv'  # Update with your CSV path
    data = pd.read_csv(csv_path)

    # Extract L, a, b from the CSV file
    L_csv = data['L'].values
    a_csv = data['a'].values
    b_csv = data['b'].values

    # Convert CSV LAB to LCH
    L_csv, C_csv, H_csv = lab_to_lch(L_csv, a_csv, b_csv)

    # Create convex hulls for both image data and CSV data
    points_image = np.column_stack((C_image, L_image))  # Combine C and L values for image
    points_csv = np.column_stack((C_csv, L_csv))        # Combine C and L values for CSV

    hull_image = ConvexHull(points_image)
    hull_csv = ConvexHull(points_csv)

    # Function to check if a point is inside the CMYK convex hull
    def is_point_in_hull(point, hull):
        """ Check if a point is inside the given convex hull """
        return np.all(np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1] <= 0)

    def map_point_to_hull_boundary(point, hull_points):
        """ Map a point outside the hull to the nearest point on the hull boundary """
        distances = np.linalg.norm(hull_points - point, axis=1)
        nearest_index = np.argmin(distances)
        return hull_points[nearest_index]

    # Extract RGB convex hull points
    rgb_hull_points = points_image[hull_image.vertices]
    # Extract CMYK convex hull boundary points
    cmyk_hull_points = points_csv[hull_csv.vertices]

    # Create a list to store the clipped RGB points
    clipped_rgb_points = []
    # Check each point in the RGB convex hull
    for point in rgb_hull_points:
        if is_point_in_hull(point, hull_csv):
            # If the point is inside the CMYK convex hull, keep it
            clipped_rgb_points.append(point)
        else:
            # If the point is outside, map it to the nearest point on the CMYK boundary
            clipped_rgb_points.append(map_point_to_hull_boundary(point, cmyk_hull_points))

    # Convert clipped RGB points to a NumPy array
    clipped_rgb_points = np.array(clipped_rgb_points)

    # Step 5: Plot the modified RGB gamut and the CMYK convex hull
    plt.figure(figsize=(8, 6))

    # Plot the original convex hull for CMYK data (dashed green lines) with the label 'Original CMYK gamut'
    csv_hull_points = np.append(hull_csv.vertices, hull_csv.vertices[0])  # Close the loop
    plt.plot(points_csv[csv_hull_points, 0], points_csv[csv_hull_points, 1], 'g--', linewidth=2, label='Fogra39 gamut')

    # Plot the modified CMYK convex hull with solid green lines (although it's unchanged)
    plt.plot(points_csv[csv_hull_points, 0], points_csv[csv_hull_points, 1], 'g-', linewidth=2)

    # Plot the original convex hull for RGB data (dashed red lines) with the label 'Original RGB gamut'
    rgb_hull_points_closed = np.append(hull_image.vertices, hull_image.vertices[0])  # Close the loop
    plt.plot(points_image[rgb_hull_points_closed, 0], points_image[rgb_hull_points_closed, 1], 'r--', linewidth=2, label='Original image gamut')

    # Plot the clipped RGB points (solid red lines) with the label 'Clipped RGB gamut'
    clipped_hull = ConvexHull(clipped_rgb_points)  # Create the new convex hull for the clipped RGB points
    clipped_hull_points = np.append(clipped_hull.vertices, clipped_hull.vertices[0])  # Ensure it's closed
    plt.plot(clipped_rgb_points[clipped_hull_points, 0], clipped_rgb_points[clipped_hull_points, 1], 'r-', linewidth=2, label='Clipped image gamut')

    # Set X-axis to start from 0
    plt.xlim(0, max(np.max(C_image), np.max(C_csv)))  # Ensure both datasets fit in the plot

    # Add labels, title, and legend
    plt.xlabel('Chroma (C)')
    plt.ylabel('Lightness (L)')
    plt.title('Image clipped to fit within the Fogra39 gamut')

    # Add legend to the top-right corner
    plt.legend(loc='upper right')
    plt.grid(True)

    # Show plot
    plt.show()

# Part 3: Clip the image to fit within the CMYK gamut
def clip_image():
    # Load the image and convert to LAB, then LCH
    image_path = 'newdataset/sample.jpg'  # Update with your image path
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR by default, so convert to RGB
    image_lab = color.rgb2lab(image_rgb)  # Convert to LAB

    # Extract L, a, b from the image
    L_image = image_lab[..., 0].flatten()
    a_image = image_lab[..., 1].flatten()
    b_image = image_lab[..., 2].flatten()

    # Convert image LAB to LCH
    L_image, C_image, H_image = lab_to_lch(L_image, a_image, b_image)

    # Load the Fogra39 data from CSV and convert to LCH
    csv_path = 'newdataset/lab.csv'  # Update with your CSV path
    data = pd.read_csv(csv_path)

    # Extract L, a, b from the CSV file
    L_csv = data['L'].values
    a_csv = data['a'].values
    b_csv = data['b'].values

    # Convert CSV LAB to LCH
    L_csv, C_csv, H_csv = lab_to_lch(L_csv, a_csv, b_csv)

    # Create convex hulls for both image data and CSV data
    points_image = np.column_stack((C_image, L_image))  # Combine C and L values for image
    points_csv = np.column_stack((C_csv, L_csv))        # Combine C and L values for CSV

    hull_csv = ConvexHull(points_csv)

    # Function to check if a point is inside the CMYK convex hull
    def is_point_in_hull(point, hull):
        """ Check if a point is inside the given convex hull """
        return np.all(np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1] <= 0)

    def map_point_to_hull_boundary(point, hull_points):
        """ Map a point outside the hull to the nearest point on the hull boundary """
        distances = np.linalg.norm(hull_points - point, axis=1)
        nearest_index = np.argmin(distances)
        return hull_points[nearest_index]

    # Extract CMYK convex hull boundary points
    cmyk_hull_points = points_csv[hull_csv.vertices]

    # Clip the entire image's LCH values based on CMYK convex hull
    clipped_L_image, clipped_C_image, clipped_H_image = [], [], []
    for L, C, H in zip(L_image, C_image, H_image):
        point = np.array([C, L])
        if is_point_in_hull(point, hull_csv):
            # If the point is inside the CMYK convex hull, keep it
            clipped_L_image.append(L)
            clipped_C_image.append(C)
            clipped_H_image.append(H)
        else:
            # If the point is outside, map it to the nearest point on the CMYK boundary
            nearest_point = map_point_to_hull_boundary(point, cmyk_hull_points)
            clipped_L_image.append(nearest_point[1])  # L value
            clipped_C_image.append(nearest_point[0])  # C value
            clipped_H_image.append(0)  # For simplicity, set H to 0 (this can be improved)

    # Convert the lists back to arrays
    clipped_L_image = np.array(clipped_L_image)
    clipped_C_image = np.array(clipped_C_image)
    clipped_H_image = np.array(clipped_H_image)

    # Convert clipped LCH back to LAB, and then to RGB
    clipped_Lab_L, clipped_Lab_a, clipped_Lab_b = lch_to_lab(clipped_L_image, clipped_C_image, clipped_H_image)

    # Reshape back to original image shape
    clipped_Lab_L = clipped_Lab_L.reshape(image_lab.shape[:2])
    clipped_Lab_a = clipped_Lab_a.reshape(image_lab.shape[:2])
    clipped_Lab_b = clipped_Lab_b.reshape(image_lab.shape[:2])

    # Reconstruct the clipped LAB image
    clipped_lab_image = np.stack([clipped_Lab_L, clipped_Lab_a, clipped_Lab_b], axis=2)

    # Convert clipped LAB to RGB
    clipped_rgb_image = color.lab2rgb(clipped_lab_image)

    # Display the original and clipped images side by side
    plt.figure(figsize=(16, 8))  # Make the figure wider for side-by-side display

    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # Display clipped image
    plt.subplot(1, 2, 2)
    plt.imshow(clipped_rgb_image)
    plt.title('Gamut Clipped Image (Nearest Neighbour)')
    plt.axis('off')

    # Show the combined plot
    plt.show()

# Run all the steps sequentially
plot_gamuts()
gamut_mapping()
clip_image()
