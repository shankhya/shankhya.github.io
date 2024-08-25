#read scanned image
import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def perspective_transform(image, pts, target_size):
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [target_size[0] - 1, 0],
        [target_size[0] - 1, target_size[1] - 1],
        [0, target_size[1] - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, target_size)

    return warped


def detect_fiducial_markers(image, marker_size=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) > 5:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            area = cv2.contourArea(contour)
            if radius > marker_size * 0.5 and area > 3.14 * (marker_size * 0.5) ** 2:
                markers.append((int(x), int(y)))  # Add as a tuple of integers

    if len(markers) != 4:
        raise ValueError("Could not detect all four fiducial markers.")

    ordered_markers = order_points(np.array(markers))

    return ordered_markers


def segment_patches_and_extract_rgb(scanned_image_path, rows, cols, target_size=(600, 400), marker_size=10, padding=15,
                                    margin=5):
    image = cv2.imread(scanned_image_path)
    if image is None:
        raise ValueError(f"Cannot load image from path: {scanned_image_path}")

    print(f"Image dimensions: {image.shape[:2]}")

    markers = detect_fiducial_markers(image, marker_size)
    warped = perspective_transform(image, markers, target_size)

    patch_width = (target_size[0] - 2 * marker_size - (cols - 1) * padding) // cols
    patch_height = (target_size[1] - 2 * marker_size - (rows - 1) * padding) // rows

    r_values = []
    g_values = []
    b_values = []

    for row in range(rows):
        for col in range(cols):
            # Adjust the position to encompass the patch area without including white buffer zones
            top_left_x = col * (patch_width + padding) + marker_size + margin
            top_left_y = row * (patch_height + padding) + marker_size + margin
            bottom_right_x = top_left_x + patch_width - 2 * margin
            bottom_right_y = top_left_y + patch_height - 2 * margin

            # Draw rectangles around the patches for visual verification
            cv2.rectangle(warped, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

            # Crop the patch from the image
            patch = warped[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Separate the channels
            mean_b, mean_g, mean_r = cv2.mean(patch)[:3]

            r_values.append(mean_r)
            g_values.append(mean_g)
            b_values.append(mean_b)

    debug_image_path = "debug_warped_image.png"
    cv2.imwrite(debug_image_path, warped)
    print(f"Debug image with rectangles saved as {debug_image_path}")

    return r_values, g_values, b_values


scanned_image_path = input("Enter the path to the scanned test target image: ")
rows = int(input("Enter the number of rows in the target: "))
cols = int(input("Enter the number of columns in the target: "))

r_values, g_values, b_values = segment_patches_and_extract_rgb(scanned_image_path, rows, cols)

for i in range(len(r_values)):
    print(f"Patch {i + 1}: R = {r_values[i]}, G = {g_values[i]}, B = {b_values[i]}")
