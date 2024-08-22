import numpy as np
from PIL import Image, ImageCms
from skimage import color, img_as_float
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage import io
def generate_adobe_rgb_points(profile_path, num_points=10):
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
def generate_gbd_from_lab_points(lab_points):
    hull = ConvexHull(lab_points)
    return hull
def scl_clip(color_lab, gbd):
    # Map towards L* = 50
    l_ref = 50.0
    direction = color_lab - np.array([l_ref, 0, 0])
    scaled_direction = direction / np.linalg.norm(direction)
    # Determine the distance to move the color towards the L* = 50 line
    distances = gbd.equations @ np.append(color_lab, 1)
    distance_to_move = np.min(distances[distances > 0], initial=0)
    clipped_color = color_lab - distance_to_move * scaled_direction
    return clipped_color
def c_clip(color_lab, gbd):
    if not np.all(gbd.equations @ np.append(color_lab, 1) <= 0):
        clipped_color = color_lab.copy()
        clipped_color[1:] *= 0.95  # Reducing chroma
        return clipped_color
    return color_lab
def l_clip(color_lab, gbd):
    if not np.all(gbd.equations @ np.append(color_lab, 1) <= 0):
        clipped_color = color_lab.copy()
        clipped_color[0] = max(0, clipped_color[0] - 10)  # Reducing lightness
        return clipped_color
    return color_lab
def map_colors(img_lab, gbd, method):
    mapped_colors = np.zeros_like(img_lab)
    for i in range(img_lab.shape[0]):
        for j in range(img_lab.shape[1]):
            if method == 'SCLIP':
                mapped_colors[i, j] = scl_clip(img_lab[i, j], gbd)
            elif method == 'CCLIP':
                mapped_colors[i, j] = c_clip(img_lab[i, j], gbd)
            elif method == 'LCLIP':
                mapped_colors[i, j] = l_clip(img_lab[i, j], gbd)
    return mapped_colors
def plot_gamut_slice(img_lab, mapped_labs, titles):
    fig, axes = plt.subplots(1, len(mapped_labs)+1, figsize=(20, 10))
    # Plot original gamut
    L = img_lab[:, :, 0].flatten()
    C = np.sqrt(img_lab[:, :, 1].flatten() ** 2 + img_lab[:, :, 2].flatten() ** 2)
    axes[0].scatter(C, L, c=color.lab2rgb(img_lab).reshape(-1, 3), s=1)
    axes[0].set_xlabel('C* (Chroma)')
    axes[0].set_ylabel('L* (Lightness)')
    axes[0].set_title('Original Gamut in L/C Space')
    axes[0].set_xlim([0, max(C)])
    axes[0].set_ylim([0, 100])
    for ax, mapped_lab, title in zip(axes[1:], mapped_labs, titles):
        L = mapped_lab[:, :, 0].flatten()
        C = np.sqrt(mapped_lab[:, :, 1].flatten() ** 2 + mapped_lab[:, :, 2].flatten() ** 2)
        ax.scatter(C, L, c=color.lab2rgb(mapped_lab).reshape(-1, 3), s=1)
        ax.set_xlabel('C* (Chroma)')
        ax.set_ylabel('L* (Lightness)')
        ax.set_title(title)
        ax.set_xlim([0, max(C)])
        ax.set_ylim([0, 100])
    plt.savefig('gamut_lc_comparison_all_clips.png')
    plt.close()
def save_comparison_images(original_rgb, mapped_rgbs, filenames):
    fig, axes = plt.subplots(1, len(mapped_rgbs) + 1, figsize=(20, 10))
    # Plot the original image
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image')
    # Plot the mapped images with annotations
    for ax, mapped_rgb, filename in zip(axes[1:], mapped_rgbs, filenames):
        ax.imshow(mapped_rgb)
        ax.set_title(filename.replace('.png', ''))
    plt.savefig('image_comparison_annotated.png')
    plt.close()
# Load your image
img = Image.open('sample.jpg')
# Convert to LAB color space
img_rgb = img_as_float(img)
img_lab = color.rgb2lab(img_rgb)
# Generate AdobeRGB LAB points using the ICC profile
adobe_rgb_icc_profile = 'AdobeRGB1998.icc'  # Ensure this file is in your working directory
lab_points = generate_adobe_rgb_points(adobe_rgb_icc_profile)
# Generate the GBD using Convex Hull
gbd = generate_gbd_from_lab_points(lab_points)
# Map colors using different clipping methods
mapped_labs = [map_colors(img_lab, gbd, method) for method in ['SCLIP', 'CCLIP', 'LCLIP']]
mapped_rgbs = [color.lab2rgb(mapped_lab) for mapped_lab in mapped_labs]
# Plot the gamut slices
plot_gamut_slice(img_lab, mapped_labs,
                 titles=['SCLIP Gamut in L/C Space', 'CCLIP Gamut in L/C Space', 'LCLIP Gamut in L/C Space'])
# Save a side-by-side comparison of the original and mapped images with annotations
save_comparison_images(img_rgb, mapped_rgbs, ['SCLIP Image', 'CCLIP Image', 'LCLIP Image'])