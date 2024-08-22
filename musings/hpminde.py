import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image, ImageCms
from skimage import color, img_as_float
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage import io
# Function to generate LAB points for AdobeRGB using ICC profile
def generate_adobe_rgb_points(profile_path, num_points=50): #125,000 grid points are generated. Higher values will yield more grid points but computationally intensive
    rgb_grid = np.linspace(0, 1, num_points)
    r, g, b = np.meshgrid(rgb_grid, rgb_grid, rgb_grid)
    rgb_points = np.c_[r.flatten(), g.flatten(), b.flatten()]
    srgb_profile = ImageCms.createProfile('sRGB')
    adobe_rgb_profile = ImageCms.getOpenProfile(profile_path)
    pil_image = Image.fromarray((rgb_points * 255).astype('uint8').reshape((num_points ** 3, 1, 3)), 'RGB')
    transform = ImageCms.buildTransform(srgb_profile, adobe_rgb_profile, "RGB", "RGB")
    adobe_rgb_image = ImageCms.applyTransform(pil_image, transform)
    adobe_rgb_points = np.array(adobe_rgb_image).reshape(-1, 3) / 255.0
    lab_points = color.rgb2lab(adobe_rgb_points)
    return lab_points
# Function to generate Gamut Boundary Descriptor (GBD) using Convex Hull
def generate_gbd_from_lab_points(lab_points):
    hull = ConvexHull(lab_points)
    return hull
# HPMINDE algorithm for clipping colors to the gamut
def hpmin_de_clip(color_lab, gbd_points):
    chroma = np.sqrt(color_lab[1] ** 2 + color_lab[2] ** 2)
    hue_angle = np.arctan2(color_lab[2], color_lab[1])
    chromas = np.sqrt(gbd_points[:, 1] ** 2 + gbd_points[:, 2] ** 2)
    hue_angles = np.arctan2(gbd_points[:, 2], gbd_points[:, 1])
    hue_diffs = np.abs(hue_angle - hue_angles)
    weighted_chroma_diffs = (chroma - chromas) ** 2
    weighted_hue_diffs = (hue_diffs * chroma) ** 2
    weighted_delta_es = np.sqrt((color_lab[0] - gbd_points[:, 0]) ** 2 + weighted_chroma_diffs + weighted_hue_diffs)
    closest_point = gbd_points[np.argmin(weighted_delta_es)]
    return closest_point
# Function to map colors to fit within the AdobeRGB space using HPMINDE method
def map_colors(img_lab, gbd):
    gbd_points = gbd.points
    mapped_colors = np.apply_along_axis(hpmin_de_clip, 2, img_lab, gbd_points=gbd_points)
    return mapped_colors
# Function to plot the original and mapped gamuts in L/C space
def plot_gamut_slice(img_lab, mapped_lab):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    L = img_lab[:, :, 0].flatten()
    C = np.sqrt(img_lab[:, :, 1].flatten() ** 2 + img_lab[:, :, 2].flatten() ** 2)
    ax[0].scatter(C, L, c=color.lab2rgb(img_lab).reshape(-1, 3), s=1)
    ax[0].set_xlabel('C* (Chroma)')
    ax[0].set_ylabel('L* (Lightness)')
    ax[0].set_title('Original Gamut in L/C Space')
    ax[0].set_xlim([0, max(C)])
    ax[0].set_ylim([0, 100])
    L_mapped = mapped_lab[:, :, 0].flatten()
    C_mapped = np.sqrt(mapped_lab[:, :, 1].flatten() ** 2 + mapped_lab[:, :, 2].flatten() ** 2)
    ax[1].scatter(C_mapped, L_mapped, c=color.lab2rgb(mapped_lab).reshape(-1, 3), s=1)
    ax[1].set_xlabel('C* (Chroma)')
    ax[1].set_ylabel('L* (Lightness)')
    ax[1].set_title('Mapped Gamut in L/C Space')
    ax[1].set_xlim([0, max(C_mapped)])
    ax[1].set_ylim([0, 100])
    plt.savefig('gamut_lc_comparison_hpmin_de_optimized.png')
    plt.close(fig)  # Close the figure after saving
def plot_image_comparison(original_rgb, mapped_rgb, filename):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(original_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    ax[1].imshow(mapped_rgb)
    ax[1].set_title("HPMINDE Mapped Image")
    ax[1].axis('off')
    plt.savefig(filename)
    plt.close(fig)  # Close the figure after saving
# Load and process the image
img = Image.open('sample1.jpg')
# Convert to LAB color space
img_rgb = img_as_float(img)
img_lab = color.rgb2lab(img_rgb)
# Generate AdobeRGB LAB points using the ICC profile
adobe_rgb_icc_profile = 'AdobeRGB1998.icc'
lab_points = generate_adobe_rgb_points(adobe_rgb_icc_profile)
# Generate the GBD using Convex Hull
gbd = generate_gbd_from_lab_points(lab_points)
# Map colors using HPMINDE method
mapped_lab = map_colors(img_lab, gbd)
# Convert mapped LAB back to RGB
mapped_rgb = color.lab2rgb(mapped_lab)
# Plot the original and mapped gamuts in L/C space
plot_gamut_slice(img_lab, mapped_lab)
# Plot a side-by-side comparison of the original and mapped images
plot_image_comparison(img_rgb, mapped_rgb, 'hpmin_de_image_comparison_optimized.jpg')