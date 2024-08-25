##generate target
import csv
from PIL import Image, ImageDraw
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color


def lab_to_rgb(lab):
    lab_color = LabColor(lab[0], lab[1], lab[2])
    rgb_color = convert_color(lab_color, sRGBColor)
    # Convert to 8-bit integer values
    rgb_tuple = rgb_color.get_value_tuple()
    rgb_8bit = [min(max(int(x * 255), 0), 255) for x in rgb_tuple]
    return tuple(rgb_8bit)


def add_fiducial_markers(draw, image_width, image_height, marker_size=10):
    marker_positions = [
        (marker_size, marker_size),  # Top-left
        (image_width - marker_size - 1, marker_size),  # Top-right
        (image_width - marker_size - 1, image_height - marker_size - 1),  # Bottom-right
        (marker_size, image_height - marker_size - 1)  # Bottom-left
    ]

    for (x, y) in marker_positions:
        draw.ellipse((x - marker_size, y - marker_size, x + marker_size, y + marker_size), fill="black")


def generate_test_target(lab_values, rows, cols, patch_width=100, patch_height=100, padding=15, marker_size=10):
    image_width = cols * (patch_width + padding) + padding
    image_height = rows * (patch_height + padding) + padding

    image_width += 2 * marker_size
    image_height += 2 * marker_size

    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)

    add_fiducial_markers(draw, image_width, image_height, marker_size)

    for i, lab in enumerate(lab_values):
        row = i // cols
        col = i % cols
        rgb = lab_to_rgb(lab)

        top_left_x = col * (patch_width + padding) + padding + marker_size
        top_left_y = row * (patch_height + padding) + padding + marker_size
        bottom_right_x = top_left_x + patch_width
        bottom_right_y = top_left_y + patch_height

        draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], fill=rgb)

    return image


csv_file_path = input("Enter the path to the CSV file with LAB values: ")

lab_values = []
with open(csv_file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        L = float(row['L'])
        a = float(row['a'])
        b = float(row['b'])
        lab_values.append((L, a, b))

rows = int(input("Enter the number of rows: "))
cols = int(input("Enter the number of columns: "))

test_target_image = generate_test_target(lab_values, rows, cols)
output_path = input("Enter the output path for the test target image (e.g., output.png): ")
test_target_image.save(output_path)
test_target_image.show()
