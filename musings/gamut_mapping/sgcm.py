import numpy as np
import pandas as pd
from skimage import io, color
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import ConvexHull

# *** Step 1: Load the Image ***
image = io.imread('lena.png')  # Replace with your image path

# *** Step 2: Convert the Image to CIELAB Color Space ***
lab_image = color.rgb2lab(image)
lab_colors = lab_image.reshape(-1, 3)

# *** Step 3: Load and Process the CMYK Printer Gamut Data ***
cmyk_lab_data = pd.read_csv('lab.csv', header=None, dtype=str, delimiter=',', engine='python')

# Handle decimal commas and convert to numeric
cmyk_lab_data = cmyk_lab_data.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
cmyk_lab_data = cmyk_lab_data.apply(pd.to_numeric, errors='coerce')
cmyk_lab_data.dropna(inplace=True)
lab_cmyk = cmyk_lab_data.values.astype(np.float64)

if lab_cmyk.shape[1] != 3:
    raise ValueError("CSV file must have exactly three columns for L*, a*, and b* values.")

# *** Step 4: Build Gamut Boundary Descriptions ***
# Convert CMYK gamut Lab values to LCH
def lab_to_lch(lab):
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    C = np.sqrt(a**2 + b**2)
    h = np.arctan2(b, a)  # Hue in radians
    h_deg = (np.degrees(h) + 360) % 360  # Convert to degrees and ensure positive values
    return np.column_stack((L, C, h_deg))

lab_cmyk_lch = lab_to_lch(lab_cmyk)

# *** Step 5: Compute Gamut Boundaries ***
hue_angles = np.linspace(0, 360, 360)
max_C_at_hue = np.zeros_like(hue_angles)
max_L_at_hue = np.zeros_like(hue_angles)
min_L_at_hue = np.zeros_like(hue_angles)

for i, hue in enumerate(hue_angles):
    hue_diff = np.abs(lab_cmyk_lch[:, 2] - hue)
    hue_diff = np.minimum(hue_diff, 360 - hue_diff)
    idx = hue_diff < 3  # 3-degree tolerance
    if np.any(idx):
        max_C_at_hue[i] = np.max(lab_cmyk_lch[idx, 1])
        max_L_at_hue[i] = np.max(lab_cmyk_lch[idx, 0])
        min_L_at_hue[i] = np.min(lab_cmyk_lch[idx, 0])
    else:
        max_C_at_hue[i] = 0
        max_L_at_hue[i] = 100
        min_L_at_hue[i] = 0

# Interpolate functions for maximum chroma and lightness boundaries
max_C_func = interp1d(hue_angles, max_C_at_hue, kind='linear', fill_value='extrapolate')
max_L_func = interp1d(hue_angles, max_L_at_hue, kind='linear', fill_value='extrapolate')
min_L_func = interp1d(hue_angles, min_L_at_hue, kind='linear', fill_value='extrapolate')

# *** Step 6: Apply Sigmoidal Compression ***
lab_colors_lch = lab_to_lch(lab_colors)
L_img = lab_colors_lch[:, 0]
C_img = lab_colors_lch[:, 1]
h_img = lab_colors_lch[:, 2]

# Compute maximum allowable chroma and lightness for each pixel's hue
max_C_allowed = max_C_func(h_img)
max_L_allowed = max_L_func(h_img)
min_L_allowed = min_L_func(h_img)

def sigmoidal_compression(x, x_min, x_max, compression_strength=0.67):
    x_norm = (x - x_min) / (x_max - x_min) * 2 - 1
    y = x_norm / (1 + compression_strength * np.abs(x_norm))
    y_mapped = (y + 1) / 2 * (x_max - x_min) + x_min
    return y_mapped

# Compress lightness
L_img_compressed = sigmoidal_compression(L_img, min_L_allowed, max_L_allowed)

# Compress chroma
C_img_compressed = C_img.copy()
out_of_gamut = C_img > max_C_allowed
C_img_compressed[out_of_gamut] = max_C_allowed[out_of_gamut] * (
    C_img[out_of_gamut] / C_img[out_of_gamut].max()
)
C_img_compressed[~out_of_gamut] = sigmoidal_compression(C_img[~out_of_gamut], 0, max_C_allowed[~out_of_gamut])

# *** Step 7: Reconstruct Lab Colors ***
a_img_compressed = C_img_compressed * np.cos(np.radians(h_img))
b_img_compressed = C_img_compressed * np.sin(np.radians(h_img))
lab_colors_compressed = np.column_stack((L_img_compressed, a_img_compressed, b_img_compressed))

# *** Step 8: Reconstruct and Save the Compressed Image ***
lab_image_compressed = lab_colors_compressed.reshape(lab_image.shape)
rgb_image_compressed = color.lab2rgb(lab_image_compressed)
rgb_image_compressed = np.clip(rgb_image_compressed, 0, 1)
rgb_image_compressed_uint8 = (rgb_image_compressed * 255).astype(np.uint8)

# *** Step 9: Plot Before and After Images ***
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# Plot the original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Plot the compressed image
axes[1].imshow(rgb_image_compressed_uint8)
axes[1].set_title("Compressed Image")
axes[1].axis('off')

plt.tight_layout()
plt.show()

# *** Step 10: Compute Convex Hulls ***
hull_image_original_sampled = ConvexHull(lab_colors)
hull_image_compressed_sampled = ConvexHull(lab_colors_compressed)
hull_cmyk = ConvexHull(lab_cmyk)

# *** Step 11: Calculate Gamut Volumes ***
volume_image_original = hull_image_original_sampled.volume
volume_image_compressed = hull_image_compressed_sampled.volume
volume_cmyk = hull_cmyk.volume

# *** Step 12: Plot All Three Gamuts ***
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# ** Plot the CMYK Printer Gamut as a Colored Wireframe **
for simplex in hull_cmyk.simplices:
    triangle_lab = lab_cmyk[simplex]
    triangle_rgb = rgb_cmyk[simplex]
    avg_rgb = np.mean(triangle_rgb, axis=0)
    lines = [
        [triangle_lab[0], triangle_lab[1]],
        [triangle_lab[1], triangle_lab[2]],
        [triangle_lab[2], triangle_lab[0]]
    ]
    line_collection = Line3DCollection(lines, colors=[avg_rgb], linewidths=1.0)
    ax.add_collection3d(line_collection)
    poly = Poly3DCollection([triangle_lab], alpha=0.1)
    poly.set_facecolor(avg_rgb)
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)

# ** Plot the Original Image Gamut as a Grayscale Solid **
for simplex in hull_image_original_sampled.simplices:
    triangle_lab = lab_colors[simplex]
    triangle_gray = gray_colors_sampled[simplex]
    poly = Poly3DCollection([triangle_lab], alpha=0.5)
    poly.set_facecolor(np.mean(triangle_gray, axis=0))
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)

# ** Plot the Compressed Image Gamut as a Solid with Actual Colors **
for simplex in hull_image_compressed_sampled.simplices:
    triangle_lab = lab_colors_compressed[simplex]
    triangle_rgb = rgb_colors_compressed_sampled[simplex]
    poly = Poly3DCollection([triangle_lab], alpha=0.8)
    poly.set_facecolor(np.mean(triangle_rgb, axis=0))
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)

# Set Axis Labels and Limits
ax.set_xlabel('L*')
ax.set_ylabel('a*')
ax.set_zlabel('b*')
ax.set_xlim(0, 100)
ax.set_ylim(-128, 127)
ax.set_zlim(-128, 127)
ax.view_init(elev=30, azim=30)

# Hide Grid Lines and Axis Ticks
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# *** Step 13: Add Gamut Volumes Below Plot in Single Line ***
volume_text = (
    f'CMYK Gamut Volume: {volume_cmyk:.2f} | '
    f'Original Image Gamut Volume: {volume_image_original:.2f} | '
    f'Compressed Image Gamut Volume: {volume_image_compressed:.2f}'
)
plt.figtext(0.5, 0.01, volume_text, wrap=True, horizontalalignment='center', fontsize=12)

# *** Step 14: Add Legend to the Top Right ***
legend_elements = [
    Line2D([0], [0], color='k', lw=2, label='CMYK Printer Gamut Wireframe'),
    Line2D([0], [0], color='gray', marker='s', markersize=10, markerfacecolor='gray', alpha=0.5, label='Original Image Gamut'),
    Line2D([0], [0], color='none', marker='s', markersize=10, markerfacecolor='blue', alpha=0.8, label='Compressed Image Gamut')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()
