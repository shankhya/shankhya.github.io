# SCIELAB-like local color-difference demo

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

# import skimage for color conversions and sample images
has_skimage = True
try:
    from skimage import data, color, util, img_as_float, img_as_ubyte
except Exception as e:
    has_skimage = False
    print("skimage not available; falling back to basic sRGB->Lab approximations.")

# import ciede2000 implementation (skimage has deltaE_ciede2000)
has_ciede2000 = True
try:
    from skimage.color import deltaE_ciede2000
except Exception as e:
    has_ciede2000 = False
    print("CIEDE2000 function not available; will use CIE76 (simple Euclidean Lab) as fallback.")


def srgb_to_lab(img):
    # img assumed float in [0,1], shape HxWx3
    if has_skimage:
        return color.rgb2lab(img)
    else:
       
        def srgb_to_linear(rgb):
            mask = rgb <= 0.04045
            lin = np.where(mask, rgb/12.92, ((rgb+0.055)/1.055)**2.4)
            return lin
        def linear_srgb_to_xyz(rgb_lin):
            M = np.array([[0.4124564,0.3575761,0.1804375],
                          [0.2126729,0.7151522,0.0721750],
                          [0.0193339,0.1191920,0.9503041]])
            xyz = rgb_lin @ M.T
            return xyz * 100.0
        def xyz_to_lab(xyz):
            Xn, Yn, Zn = 95.047, 100.0, 108.883
            xyz_scaled = np.stack([xyz[:,:,0]/Xn, xyz[:,:,1]/Yn, xyz[:,:,2]/Zn], axis=2)
            delta = 6/29
            def f(t):
                return np.where(t > delta**3, np.cbrt(t), (t/(3*delta**2)) + 4/29)
            fxyz = f(xyz_scaled)
            L = 116 * fxyz[:,:,1] - 16
            a = 500 * (fxyz[:,:,0] - fxyz[:,:,1])
            b = 200 * (fxyz[:,:,1] - fxyz[:,:,2])
            lab = np.stack([L,a,b], axis=2)
            return lab
        rgb_lin = srgb_to_linear(img)
        xyz = linear_srgb_to_xyz(rgb_lin)
        return xyz_to_lab(xyz)

def delta_e_map(lab_ref, lab_test):
    # lab_ref/test shape HxWx3, L in 0..100
    if has_ciede2000 and has_skimage:
        # skimage expects Lab images in same range
        return deltaE_ciede2000(lab_ref, lab_test)
    else:
        # simple Euclidean distance
        diff = lab_ref - lab_test
        return np.linalg.norm(diff, axis=2)

# Create sample image
if has_skimage:
    img_rgb = img_as_float(data.astronaut())  # float in 0..1
    img_rgb = img_rgb[..., :3]  # ensure 3 channels
else:
    # fallback: create synthetic colorful gradient
    h, w = 256, 512
    x = np.linspace(0,1,w)
    y = np.linspace(0,1,h)
    xv, yv = np.meshgrid(x,y)
    img_rgb = np.stack([xv, yv, 0.5 + 0.5*xv*yv], axis=2)
    img_rgb = np.clip(img_rgb, 0, 1)

# Create two distorted variants
def hue_shift(img, deg_shift=15):
    # Convert to HSV using skimage if available, else simple rgb->hsv implementation
    if has_skimage:
        hsv = color.rgb2hsv(img)
        hsv[...,0] = (hsv[...,0] + deg_shift/360.0) % 1.0
        shifted = color.hsv2rgb(hsv)
        return shifted
    else:
        # naive conversion using matplotlib.colors
        import matplotlib.colors as mcolors
        flat = img.reshape(-1,3)
        hsv = np.array([mcolors.rgb_to_hsv(pixel) for pixel in flat]).reshape(img.shape)
        hsv[...,0] = (hsv[...,0] + deg_shift/360.0) % 1.0
        shifted = np.array([mcolors.hsv_to_rgb(pixel) for pixel in hsv.reshape(-1,3)]).reshape(img.shape)
        return shifted

def chroma_noise(img, scale=10.0):
    # convert to Lab, add noise to a,b channels, and convert back to rgb
    lab = srgb_to_lab(img)
    lab2 = lab.copy()
    rng = np.random.RandomState(0)
    noise_a = rng.normal(scale=scale, size=lab[...,1].shape)
    noise_b = rng.normal(scale=scale, size=lab[...,2].shape)
    lab2[...,1] = lab2[...,1] + noise_a
    lab2[...,2] = lab2[...,2] + noise_b
    # convert back to rgb via skimage if available
    if has_skimage:
        rgb2 = color.lab2rgb(lab2)
        rgb2 = np.clip(rgb2, 0, 1)
        return rgb2
    else:
        # approximate lab -> xyz -> srgb
        def lab_to_xyz(lab):
            L, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
            Y = (L + 16) / 116.0
            X = a / 500.0 + Y
            Z = Y - b / 200.0
            def f_inv(t):
                delta = 6/29
                return np.where(t > delta, t**3, 3*delta**2*(t - 4/29))
            X = f_inv(X) * 95.047
            Y = f_inv(Y) * 100.0
            Z = f_inv(Z) * 108.883
            xyz = np.stack([X,Y,Z], axis=2)
            return xyz
        def xyz_to_srgb(xyz):
            xyz = xyz / 100.0
            M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                              [-0.9692660,  1.8760108,  0.0415560],
                              [ 0.0556434, -0.2040259,  1.0572252]])
            rgb_lin = xyz @ M_inv.T
            def lin_to_srgb(c):
                mask = c <= 0.0031308
                return np.where(mask, 12.92*c, 1.055*(c**(1/2.4)) - 0.055)
            rgb = lin_to_srgb(rgb_lin)
            return np.clip(rgb, 0, 1)
        xyz = lab_to_xyz(lab2)
        return xyz_to_srgb(xyz)

img_hue = hue_shift(img_rgb, deg_shift=20)
img_noise = chroma_noise(img_rgb, scale=8.0)

# Convert to Lab
lab_ref = srgb_to_lab(img_rgb)
lab_hue = srgb_to_lab(img_hue)
lab_noise = srgb_to_lab(img_noise)

# Compute pixelwise delta E maps
de_hue = delta_e_map(lab_ref, lab_hue)
de_noise = delta_e_map(lab_ref, lab_noise)

# SCIELAB-like multi-scale local difference
def scielab_like_map(lab_ref, lab_test, sigmas=[1,2,4], weights=None):
    # lab channels L (0), a (1), b (2)
    if weights is None:
        # weights should sum to 1
        weights = np.array([0.5, 0.3, 0.2])
    weights = np.array(weights) / np.sum(weights)
    H, W, _ = lab_ref.shape
    acc = np.zeros((H,W))
    for sigma, w in zip(sigmas, weights):
        # Gaussian blur each channel
        Lr = gaussian_filter(lab_ref[...,0], sigma=sigma)
        ar = gaussian_filter(lab_ref[...,1], sigma=sigma)
        br = gaussian_filter(lab_ref[...,2], sigma=sigma)
        Lt = gaussian_filter(lab_test[...,0], sigma=sigma)
        at = gaussian_filter(lab_test[...,1], sigma=sigma)
        bt = gaussian_filter(lab_test[...,2], sigma=sigma)
        de = np.sqrt((Lr-Lt)**2 + (ar-at)**2 + (br-bt)**2)
        acc += w * de
    return acc

sc_hue = scielab_like_map(lab_ref, lab_hue)
sc_noise = scielab_like_map(lab_ref, lab_noise)

# Normalize maps for visualization
def normalize_for_display(x):
    xm = x - x.min()
    if xm.max() > 0:
        xm = xm / xm.max()
    return xm

# Plot results
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

axes[0,0].imshow(img_rgb)
axes[0,0].set_title("Reference (RGB)")
axes[0,0].axis('off')
axes[0,1].imshow(img_hue)
axes[0,1].set_title("Hue-shifted")
axes[0,1].axis('off')
axes[0,2].imshow(img_noise)
axes[0,2].set_title("Chroma-noised")
axes[0,2].axis('off')

# Show deltaE maps (pixelwise)
im1 = axes[1,0].imshow(de_hue, cmap='inferno')
axes[1,0].set_title("ΔE (pixelwise) - Hue shift")
axes[1,0].axis('off')
fig.colorbar(im1, ax=axes[1,0], fraction=0.046, pad=0.01)

im2 = axes[1,1].imshow(sc_hue, cmap='inferno')
axes[1,1].set_title("SCIELAB-like map - Hue shift")
axes[1,1].axis('off')
fig.colorbar(im2, ax=axes[1,1], fraction=0.046, pad=0.01)

im3 = axes[1,2].imshow(de_noise, cmap='inferno')
axes[1,2].set_title("ΔE (pixelwise) - Chroma noise")
axes[1,2].axis('off')
fig.colorbar(im3, ax=axes[1,2], fraction=0.046, pad=0.01)

im4 = axes[1,3].imshow(sc_noise, cmap='inferno')
axes[1,3].set_title("SCIELAB-like map - Chroma noise")
axes[1,3].axis('off')
fig.colorbar(im4, ax=axes[1,3], fraction=0.046, pad=0.01)

# Also show zoomed crop to highlight local differences
H, W, _ = img_rgb.shape
ch_h, cw = H//3, W//3
crop_slice = (slice(ch_h, ch_h+100), slice(cw, cw+100))
axes[2,0].imshow(img_rgb[crop_slice])
axes[2,0].set_title("Ref crop")
axes[2,0].axis('off')
axes[2,1].imshow(img_hue[crop_slice])
axes[2,1].set_title("Hue crop")
axes[2,1].axis('off')
axes[2,2].imshow(de_hue[crop_slice], cmap='inferno')
axes[2,2].set_title("ΔE crop (hue)")
axes[2,2].axis('off')
axes[2,3].imshow(sc_hue[crop_slice], cmap='inferno')
axes[2,3].set_title("SCIELAB-like crop (hue)")
axes[2,3].axis('off')

plt.tight_layout()
out_path = "/mnt/data/scielab_demo.png"
fig.savefig(out_path)
print("Saved figure to:", out_path)

# Summary statistics table
import pandas as pd
rows = [
    {"distortion": "hue_shift", "mean_deltaE_pixelwise": float(np.mean(de_hue)), "mean_scielab_like": float(np.mean(sc_hue))},
    {"distortion": "chroma_noise", "mean_deltaE_pixelwise": float(np.mean(de_noise)), "mean_scielab_like": float(np.mean(sc_noise))}
]
df = pd.DataFrame(rows)
import caas_jupyter_tools as cjt
cjt.display_dataframe_to_user("SCIELAB Demo Summary", df)

# Provide normalized maps as images for quick download
import imageio
imageio.imsave("/mnt/data/de_hue.png", (normalize_for_display(de_hue)*255).astype(np.uint8))
imageio.imsave("/mnt/data/sc_hue.png", (normalize_for_display(sc_hue)*255).astype(np.uint8))
imageio.imsave("/mnt/data/de_noise.png", (normalize_for_display(de_noise)*255).astype(np.uint8))
imageio.imsave("/mnt/data/sc_noise.png", (normalize_for_display(sc_noise)*255).astype(np.uint8))

print("Saved additional map images: /mnt/data/de_hue.png, /mnt/data/sc_hue.png, /mnt/data/de_noise.png, /mnt/data/sc_noise.png")

