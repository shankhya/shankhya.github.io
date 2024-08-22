#Minimum Color Difference Gamut Clipping
#Target space: AdobeRGB1998.icc

from PIL import Image, ImageCms
import numpy as np
from skimage import color, img_as_float
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage import io

def generate_adobe_rgb_points(profile_path, num_points=10):  # Increase num_points for better sampling
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

def clip_color_to_gamut(color_lab, gbd):
    if not np.all(gbd.equations @ np.append(color_lab, 1) <= 0):
        distances = np.linalg.norm(gbd.points - color_lab, axis=1)
        closest_point = gbd.points[np.argmin(distances)]
        return closest_point
    return color_lab

def map_colors_to_adobergb(img_lab, gbd):
    mapped_colors = np.zeros_like(img_lab)
    for i in range(img_lab.shape[0]):
        for j in range(img_lab.shape[1]):
            mapped_colors[i, j] = clip_color_to_gamut(img_lab[i, j], gbd)
    return mapped_colors

def plot_gamut_slice(img_lab, mapped_lab):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Original Gamut in L/C Space
    L = img_lab[:, :, 0].flatten()
    C = np.sqrt(img_lab[:, :, 1].flatten()**2 + img_lab[:, :, 2].flatten()**2)
    ax[0].scatter(C, L, c=color.lab2rgb(img_lab).reshape(-1, 3), s=1)
    ax[0].set_xlabel('C* (Chroma)')
    ax[0].set_ylabel('L* (Lightness)')
    ax[0].set_title('Original Gamut in L/C Space')
    ax[0].set_xlim([0, max(C)])
    ax[0].set_ylim([0, 100])

    # Mapped Gamut in L/C Space
    L_mapped = mapped_lab[:, :, 0].flatten()
    C_mapped = np.sqrt(mapped_lab[:, :, 1].flatten()**2 + mapped_lab[:, :, 2].flatten()**2)
    ax[1].scatter(C_mapped, L_mapped, c=color.lab2rgb(mapped_lab).reshape(-1, 3), s=1)
    ax[1].set_xlabel('C* (Chroma)')
    ax[1].set_ylabel('L* (Lightness)')
    ax[1].set_title('Mapped Gamut in L/C Space')
    ax[1].set_xlim([0, max(C_mapped)])
    ax[1].set_ylim([0, 100])

    plt.savefig('gamut_lc_comparison.png')
    plt.close()

def save_comparison_image(original_rgb, mapped_rgb, filename):
    comparison = np.hstack((original_rgb, mapped_rgb))
    io.imsave(filename, (comparison * 255).astype(np.uint8))

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

# Map colors to fit within the AdobeRGB space
mapped_lab = map_colors_to_adobergb(img_lab, gbd)

# Convert mapped LAB back to RGB
mapped_rgb = color.lab2rgb(mapped_lab)

# Plot the original and mapped gamuts in 3D (optional)
# plot_gamuts(img_lab, mapped_lab)

# Plot the gamut slice in L/C space
plot_gamut_slice(img_lab, mapped_lab)

# Save a side-by-side comparison of the original and mapped images
save_comparison_image(img_rgb, mapped_rgb, 'image_comparison.png')
