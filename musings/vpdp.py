import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image, ImageCms
from skimage import color, img_as_float
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os
# Function to generate LAB points for AdobeRGB using ICC profile
def generate_adobe_rgb_points(profile_path, num_points=50):
    rgb_grid = np.linspace(0, 1, num_points)
    r, g, b = np.meshgrid(rgb_grid, rgb_grid, rgb_grid)
    rgb_points = np.c_[r.flatten(), g.flatten(), b.flatten()]
    srgb_profile = ImageCms.createProfile('sRGB')
    try:
        adobe_rgb_profile = ImageCms.getOpenProfile(profile_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"ICC profile '{profile_path}' not found. Please ensure the path is correct.")
    pil_image = Image.fromarray((rgb_points * 255).astype('uint8').reshape((num_points ** 3, 1, 3)), 'RGB')
    transform = ImageCms.buildTransform(srgb_profile, adobe_rgb_profile, "RGB", "RGB")
    adobe_rgb_image = ImageCms.applyTransform(pil_image, transform)
    adobe_rgb_points = np.array(adobe_rgb_image).reshape(-1, 3) / 255.0
    lab_points = color.rgb2lab(adobe_rgb_points.reshape(1, -1, 3)).reshape(-1, 3)
    return lab_points
# Function to generate Gamut Boundary Descriptor (GBD) using Convex Hull
def generate_gbd_from_lab_points(lab_points):
    hull = ConvexHull(lab_points)
    return hull
# Vividness Preserved (VP) Compression Algorithm
def vp_compression(color_lab, cusp_lightness, compression_factor=0.8):
    L, a, b = color_lab
    C = np.sqrt(a ** 2 + b ** 2)
    hue_angle = np.arctan2(b, a)
    L_mapped = L * compression_factor
    C_mapped = C * compression_factor
    a_mapped = C_mapped * np.cos(hue_angle)
    b_mapped = C_mapped * np.sin(hue_angle)
    return np.array([L_mapped, a_mapped, b_mapped])
# Depth Preserved (DP) Compression Algorithm
def dp_compression(color_lab, cusp_lightness, compression_factor=0.8):
    L, a, b = color_lab
    C = np.sqrt(a ** 2 + b ** 2)
    hue_angle = np.arctan2(b, a)
    L_mapped = L + (cusp_lightness - L) * (1 - compression_factor)
    L_mapped = np.minimum(L_mapped, cusp_lightness)
    C_mapped = C * compression_factor
    a_mapped = C_mapped * np.cos(hue_angle)
    b_mapped = C_mapped * np.sin(hue_angle)
    return np.array([L_mapped, a_mapped, b_mapped])
# Function to map colors using the selected GCA method
def map_colors(img_lab, method, cusp_lightness, compression_factor=0.8):
    # Vectorize the compression function for efficiency
    if method == 'VP':
        compression_func = vp_compression
    elif method == 'DP':
        compression_func = dp_compression
    else:
        raise ValueError("Unsupported method. Choose 'VP' or 'DP'.")
    # Ensure correct shape handling by processing each pixel individually
    mapped_lab = np.zeros_like(img_lab)
    for i in range(img_lab.shape[0]):
        for j in range(img_lab.shape[1]):
            mapped_lab[i, j] = compression_func(img_lab[i, j], cusp_lightness, compression_factor)
    return mapped_lab
# Function to plot all gamuts in separate subplots in L/C space
def plot_gamuts_separate_subplots(original_lab, mapped_lab_vp, mapped_lab_dp):
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    # Original Gamut
    L_original = original_lab[:, :, 0].flatten()
    C_original = np.sqrt(original_lab[:, :, 1].flatten() ** 2 + original_lab[:, :, 2].flatten() ** 2)
    ax[0].scatter(C_original, L_original, c=color.lab2rgb(original_lab).reshape(-1, 3), s=1, marker='.')
    ax[0].set_xlabel('C* (Chroma)')
    ax[0].set_ylabel('L* (Lightness)')
    ax[0].set_title('Original Gamut in L/C Space')
    ax[0].set_xlim([0, np.percentile(C_original, 99)])
    ax[0].set_ylim([0, 100])
    # VP Gamut
    L_mapped_vp = mapped_lab_vp[:, :, 0].flatten()
    C_mapped_vp = np.sqrt(mapped_lab_vp[:, :, 1].flatten() ** 2 + mapped_lab_vp[:, :, 2].flatten() ** 2)
    ax[1].scatter(C_mapped_vp, L_mapped_vp, c=color.lab2rgb(mapped_lab_vp).reshape(-1, 3), s=1, marker='.')
    ax[1].set_xlabel('C* (Chroma)')
    ax[1].set_ylabel('L* (Lightness)')
    ax[1].set_title('VP Gamut in L/C Space')
    ax[1].set_xlim([0, np.percentile(C_mapped_vp, 99)])
    ax[1].set_ylim([0, 100])
    # DP Gamut
    L_mapped_dp = mapped_lab_dp[:, :, 0].flatten()
    C_mapped_dp = np.sqrt(mapped_lab_dp[:, :, 1].flatten() ** 2 + mapped_lab_dp[:, :, 2].flatten() ** 2)
    ax[2].scatter(C_mapped_dp, L_mapped_dp, c=color.lab2rgb(mapped_lab_dp).reshape(-1, 3), s=1, marker='.')
    ax[2].set_xlabel('C* (Chroma)')
    ax[2].set_ylabel('L* (Lightness)')
    ax[2].set_title('DP Gamut in L/C Space')
    ax[2].set_xlim([0, np.percentile(C_mapped_dp, 99)])
    ax[2].set_ylim([0, 100])
    plt.tight_layout()
    plt.savefig('gamut_lc_comparison_subplots.png', dpi=300)
    plt.close(fig)
    print(f"Gamut L/C comparison plot saved as 'gamut_lc_comparison_subplots.png'.")
# Main processing function
def process_image_with_gca():
    # Define paths
    adobe_rgb_icc_profile = 'AdobeRGB1998.icc'  # Update the path if necessary
    input_image_path = 'sample1.jpg'  # Update the path to your input image
    # Check if ICC profile exists
    if not os.path.isfile(adobe_rgb_icc_profile):
        print(f"ICC profile '{adobe_rgb_icc_profile}' not found. Please provide the correct path.")
        return
    # Load and process the image
    try:
        img = Image.open(input_image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Input image '{input_image_path}' not found. Please provide the correct path.")
        return
    img_rgb = img_as_float(img)
    img_lab = color.rgb2lab(img_rgb)
    # Generate AdobeRGB LAB points using the ICC profile
    lab_points = generate_adobe_rgb_points(adobe_rgb_icc_profile)
    # Generate the GBD using Convex Hull
    gbd = generate_gbd_from_lab_points(lab_points)
    # Estimate the CUSP Lightness
    cusp_lightness = np.max(lab_points[:, 0])
    # VP Method
    mapped_lab_vp = map_colors(img_lab, 'VP', cusp_lightness, compression_factor=0.8)
    mapped_rgb_vp = color.lab2rgb(mapped_lab_vp)
    # DP Method
    mapped_lab_dp = map_colors(img_lab, 'DP', cusp_lightness, compression_factor=0.8)
    mapped_rgb_dp = color.lab2rgb(mapped_lab_dp)
    # Plot all gamuts in separate subplots
    plot_gamuts_separate_subplots(img_lab, mapped_lab_vp, mapped_lab_dp)
    # Plot all images together
    plot_all_images(img_rgb, mapped_rgb_vp, mapped_rgb_dp)
# Run the process
process_image_with_gca()
