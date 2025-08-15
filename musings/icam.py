#!/usr/bin/env python3
"""
icam_like.py

Simplified iCAM-like image appearance pipeline.

Usage:
    python icam_like.py --input path/to/image.jpg --mode bilateral --key 0.18 --gauss-sigma 16 --outdir outputs

If --input is omitted, a built-in test image is used (skimage astronaut or a synthetic gradient).
"""
import os
import argparse
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Try to import scikit-image for color conversions, denoise_bilateral and sample images
_has_skimage = True
try:
    from skimage import data, img_as_float, color, util, io as skio
except Exception:
    _has_skimage = False

# Try to import denoise_bilateral (different skimage versions use different argument names)
_has_bilateral = True
try:
    from skimage.restoration import denoise_bilateral
except Exception:
    _has_bilateral = False

# ---------------------------
# Color conversion utilities
# ---------------------------

# sRGB nonlinear <-> linear
def srgb_to_linear(srgb):
    """Convert sRGB (0..1) to linear RGB (0..1). srgb can be ndarray of shape HxWx3."""
    srgb = np.clip(srgb, 0.0, 1.0)
    mask = srgb <= 0.04045
    lin = np.where(mask, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return lin

def linear_to_srgb(lin):
    """Convert linear RGB (0..1) to sRGB (0..1)."""
    lin = np.clip(lin, 0.0, 1.0)
    mask = lin <= 0.0031308
    srgb = np.where(mask, lin * 12.92, 1.055 * (lin ** (1.0/2.4)) - 0.055)
    return np.clip(srgb, 0.0, 1.0)

# sRGB linear <-> CIEXYZ (D65)
M_srgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
M_xyz2srgb = np.linalg.inv(M_srgb2xyz)

def linear_srgb_to_xyz(rgb_lin):
    """Convert linear sRGB image (HxWx3) to XYZ (same shape)."""
    shp = rgb_lin.shape
    arr = rgb_lin.reshape(-1, 3)
    xyz = arr @ M_srgb2xyz.T
    return xyz.reshape(shp)

def xyz_to_linear_srgb(xyz):
    """Convert XYZ image (HxWx3) to linear sRGB (same shape)."""
    shp = xyz.shape
    arr = xyz.reshape(-1, 3)
    rgb_lin = arr @ M_xyz2srgb.T
    return rgb_lin.reshape(shp)

# ---------------------------
# Bradford chromatic adaptation
# ---------------------------

M_br = np.array([[ 0.8951,  0.2664, -0.1614],
                 [-0.7502,  1.7135,  0.0367],
                 [ 0.0389, -0.0685,  1.0296]])
M_br_inv = np.linalg.inv(M_br)

def chromatic_adaptation_brdf(xyz, src_white, dst_white, adapt_strength=1.0):
    """
    Bradford chromatic adaptation transform.
    xyz: HxWx3 or Nx3 (will be reshaped appropriately)
    src_white, dst_white: 3-element XYZ white vectors (e.g., src_white = [Xw,Yw,Zw])
    adapt_strength: in [0,1], 0=no adaptation, 1=full adaptation
    """
    src_LMS = M_br @ np.asarray(src_white)
    dst_LMS = M_br @ np.asarray(dst_white)
    scale = dst_LMS / (src_LMS + 1e-12)
    scale = 1.0 + adapt_strength * (scale - 1.0)

    shp = xyz.shape
    arr = xyz.reshape(-1, 3).T  # 3 x N
    LMS = M_br @ arr
    LMS_adapt = LMS * scale[:, np.newaxis]
    xyz_adapt = M_br_inv @ LMS_adapt
    return xyz_adapt.T.reshape(shp)

# ---------------------------
# Local adaptation estimation
# ---------------------------

def estimate_local_luminance(xyz, method='bilateral', bilateral_sigma_color=0.08, bilateral_sigma_spatial=7, gauss_sigma=16):
    """
    Estimate local luminance map from XYZ image.
    method: 'bilateral' or 'gaussian'
    bilateral parameters: sigma_color in intensity units (0..1 for normalized Y), sigma_spatial in pixels.
    gauss_sigma: gaussian blur sigma in pixels (fallback).
    """
    Y = xyz[..., 1]  # Y channel
    if method == 'bilateral' and _has_bilateral:
        # denoise_bilateral expects input in [0,1] for float images; normalize by max for stability
        Ymax = Y.max() if Y.max() > 0 else 1.0
        Y_norm = Y / Ymax
        # Use channel_axis=None for 2D arrays in recent scikit-image; older versions may ignore this
        try:
            Y_loc = denoise_bilateral(Y_norm, sigma_color=bilateral_sigma_color, sigma_spatial=bilateral_sigma_spatial, channel_axis=None)
        except TypeError:
            # older skimage versions may not accept channel_axis
            Y_loc = denoise_bilateral(Y_norm, sigma_color=bilateral_sigma_color, sigma_spatial=bilateral_sigma_spatial)
        Y_loc = Y_loc * Ymax
    else:
        Y_loc = gaussian_filter(Y, sigma=gauss_sigma)
    return Y_loc

# ---------------------------
# Local tone mapping & chroma-preserve
# ---------------------------

def local_tone_mapping(xyz, Y_local, key=0.18, epsilon=1e-9):
    """
    Heuristic local tone mapping:
      - compute Ls = key * (Y / meanY)
      - denom depends on local adaptation Y_local
      - Y_out = Ls / (1 + Ls/(1 + Y_local/meanY))
    This compresses luminance more where local luminance is higher relative to mean.
    """
    Y = xyz[..., 1]
    meanY = np.mean(Y) + epsilon
    Ls = key * (Y / meanY)
    denom = 1.0 + (Ls / (1.0 + (Y_local / meanY)))
    Y_out = Ls / (denom + epsilon)
    # Map back to a similar dynamic range by scaling with meanY (optional). We keep absolute scale modest.
    # Rescale so that mean(Y_out) ~ mean(Y)/meanY * key; often it's fine without scaling.
    return np.clip(Y_out, epsilon, None)

def apply_chromatic_preservation(xyz, Y_out):
    """
    Preserve chromaticity by scaling X and Z channels by ratio Y_out / Y.
    This keeps x,y chromaticity approximately constant while updating Y.
    """
    X = xyz[..., 0]
    Y = xyz[..., 1]
    Z = xyz[..., 2]
    ratio = Y_out / (Y + 1e-12)
    Xp = X * ratio
    Yp = Y_out
    Zp = Z * ratio
    xyz_mapped = np.stack([Xp, Yp, Zp], axis=-1)
    return xyz_mapped

# ---------------------------
# Pipeline wrapper
# ---------------------------

def icam_like_pipeline(img_rgb, use_bilateral=True, key=0.18, gauss_sigma=16, bilateral_params=None,
                       do_chromatic_adaptation=False, src_white=None, dst_white=None, adapt_strength=1.0):
    """
    Run the simplified iCAM-like pipeline on an sRGB image (float 0..1).
    Returns processed sRGB image (0..1) and a dict of intermediate maps.
    """
    # linearize sRGB
    rgb_lin = srgb_to_linear(img_rgb)
    # convert to XYZ
    xyz = linear_srgb_to_xyz(rgb_lin)
    # optional global chromatic adaptation (simulate illuminant change)
    if do_chromatic_adaptation and (src_white is not None) and (dst_white is not None):
        xyz = chromatic_adaptation_brdf(xyz, src_white, dst_white, adapt_strength=adapt_strength)

    # local luminance estimation
    method = 'bilateral' if use_bilateral else 'gaussian'
    if bilateral_params is None:
        bilateral_params = {'sigma_color': 0.08, 'sigma_spatial': 7}
    Y_local = estimate_local_luminance(xyz, method=method,
                                       bilateral_sigma_color=bilateral_params['sigma_color'],
                                       bilateral_sigma_spatial=bilateral_params['sigma_spatial'],
                                       gauss_sigma=gauss_sigma)
    # local tone mapping (Y_out)
    Y_out = local_tone_mapping(xyz, Y_local, key=key)
    # chroma-preserving XYZ reconstruction
    xyz_mapped = apply_chromatic_preservation(xyz, Y_out)
    # convert back to linear sRGB then nonlinear sRGB
    rgb_lin_out = xyz_to_linear_srgb(xyz_mapped)
    rgb_out = linear_to_srgb(rgb_lin_out)
    rgb_out = np.clip(rgb_out, 0.0, 1.0)
    maps = {'xyz': xyz, 'Y_local': Y_local, 'Y_out': Y_out, 'rgb_lin': rgb_lin, 'rgb_lin_out': rgb_lin_out}
    return rgb_out, maps

# ---------------------------
# Utilities: load, save, stats
# ---------------------------

def load_image(path=None):
    """Load an image as float [0,1]. If path is None, return skimage astronaut or synthetic gradient."""
    if path is not None:
        # prefer skimage.io if available for reading
        try:
            if _has_skimage:
                img = skio.imread(path)
                img = img_as_float(img)
            else:
                img = imageio.v2.imread(path)
                img = img.astype(np.float32) / 255.0
        except Exception as e:
            raise RuntimeError(f"Failed to read image '{path}': {e}")
        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] > 3:
            img = img[..., :3]
        return np.clip(img, 0.0, 1.0)
    else:
        # no input: try astronaut
        if _has_skimage:
            img = img_as_float(data.astronaut())
            img = img[..., :3]
            return np.clip(img, 0.0, 1.0)
        else:
            # synthetic gradient HxW 512x768
            h, w = 512, 768
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            xv, yv = np.meshgrid(x, y)
            img = np.dstack([xv, yv, 0.5 + 0.5 * xv * yv])
            return np.clip(img, 0.0, 1.0)

def save_image(path, img):
    """Save image (float 0..1 or uint8) to disk using imageio."""
    arr = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    imageio.imsave(path, arr)

def image_stats(img_rgb):
    """Return mean luminance and normalized std contrast in XYZ Y channel."""
    lin = srgb_to_linear(img_rgb)
    xyz = linear_srgb_to_xyz(lin)
    Y = xyz[..., 1]
    meanY = float(Y.mean())
    contrast = float(Y.std() / (meanY + 1e-12))
    return meanY, contrast

# ---------------------------
# Main: CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="Run a simplified iCAM-like image appearance pipeline.")
    p.add_argument("--input", "-i", help="Path to input image (optional). If omitted, demo image is used.", default=None)
    p.add_argument("--mode", choices=['bilateral', 'gaussian', 'both'], default='both',
                   help="Which local adaptation method(s) to run. 'both' runs bilateral and gaussian variants.")
    p.add_argument("--key", type=float, default=0.18, help="Key value for tone mapping (default 0.18).")
    p.add_argument("--gauss-sigma", type=float, default=16.0, help="Gaussian blur sigma for local luminance (fallback).")
    p.add_argument("--bilateral-sigma-color", type=float, default=0.08, help="Bilateral sigma_color for Y normalization.")
    p.add_argument("--bilateral-sigma-spatial", type=float, default=7.0, help="Bilateral sigma_spatial in pixels.")
    p.add_argument("--outdir", "-o", default="icam_outputs", help="Directory to save outputs.")
    p.add_argument("--no-adapt", dest='do_adapt', action='store_false', help="Disable optional global chromatic adaptation (default enabled if whitepoints provided).")
    args = p.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print("Loading input image...")
    img = load_image(args.input)
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")

    # Optional chromatic adaptation targets (disabled by default). If you want to enable adaptation,
    # set src_white and dst_white to appropriate XYZ white values. Here we keep it disabled by default.
    do_chromatic_adaptation = False
    src_white = None
    dst_white = None
    adapt_strength = 1.0

    # Run pipelines
    variants = []
    if args.mode in ('bilateral', 'both'):
        print("Running bilateral local-adaptation pipeline...")
        out_bilat, maps_bilat = icam_like_pipeline(img, use_bilateral=True,
                                                   key=args.key,
                                                   gauss_sigma=args.gauss_sigma,
                                                   bilateral_params={'sigma_color': args.bilateral_sigma_color,
                                                                     'sigma_spatial': args.bilateral_sigma_spatial},
                                                   do_chromatic_adaptation=do_chromatic_adaptation,
                                                   src_white=src_white, dst_white=dst_white,
                                                   adapt_strength=adapt_strength)
        variants.append(('bilateral', out_bilat, maps_bilat))
    if args.mode in ('gaussian', 'both'):
        print("Running gaussian local-adaptation pipeline...")
        out_gauss, maps_gauss = icam_like_pipeline(img, use_bilateral=False,
                                                   key=args.key,
                                                   gauss_sigma=args.gauss_sigma,
                                                   bilateral_params={'sigma_color': args.bilateral_sigma_color,
                                                                     'sigma_spatial': args.bilateral_sigma_spatial},
                                                   do_chromatic_adaptation=do_chromatic_adaptation,
                                                   src_white=src_white, dst_white=dst_white,
                                                   adapt_strength=adapt_strength)
        variants.append(('gaussian', out_gauss, maps_gauss))

    # Save outputs and a comparison figure
    print("Saving outputs...")
    save_image(os.path.join(outdir, "original.png"), img)
    fig, axes = plt.subplots(2, max(3, len(variants)+1), figsize=(5*(len(variants)+1), 8))
    # Top row: images
    axes[0,0].imshow(img); axes[0,0].set_title('Original (sRGB)'); axes[0,0].axis('off')
    for i, (name, out_img, maps) in enumerate(variants, start=1):
        axes[0,i].imshow(out_img); axes[0,i].set_title(f'iCAM-like ({name})'); axes[0,i].axis('off')
    # If fewer variants than columns, hide remaining axes
    for j in range(len(variants)+1, axes.shape[1]):
        axes[0,j].axis('off')

    # Bottom row: show Y_local and Y_out for the first variant (if present)
    if len(variants) > 0:
        # show Y_local and Y_out for the first variant and Y_local for second variant if exists
        name0, _, maps0 = variants[0]
        Yloc0 = maps0['Y_local']; Yout0 = maps0['Y_out']
        def norm_map(m):
            mm = m - np.nanmin(m)
            if np.nanmax(mm) > 0:
                mm = mm / np.nanmax(mm)
            return mm
        axes[1,0].imshow(norm_map(Yloc0), cmap='gray'); axes[1,0].set_title(f'Local luminance ({name0})'); axes[1,0].axis('off')
        axes[1,1].imshow(norm_map(Yout0), cmap='inferno'); axes[1,1].set_title(f'Mapped Y_out ({name0})'); axes[1,1].axis('off')
        if len(variants) > 1:
            name1, _, maps1 = variants[1]
            axes[1,2].imshow(norm_map(maps1['Y_local']), cmap='gray'); axes[1,2].set_title(f'Local luminance ({name1})'); axes[1,2].axis('off')
        # hide any leftover axes
        for j in range(3, axes.shape[1]):
            axes[1,j].axis('off')
    else:
        # no variants ran; hide bottom row
        for j in range(axes.shape[1]):
            axes[1,j].axis('off')

    fig.tight_layout()
    fig_path = os.path.join(outdir, "icam_comparison.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved comparison figure to: {fig_path}")

    # Save individual variant images
    for name, out_img, maps in variants:
        save_image(os.path.join(outdir, f"icam_processed_{name}.png"), out_img)

    # Print simple stats and save as text
    rows = []
    orig_meanY, orig_contrast = image_stats(img)
    rows.append(('original', orig_meanY, orig_contrast))
    for name, out_img, _ in variants:
        meanY, contrast = image_stats(out_img)
        rows.append((name, meanY, contrast))
    stats_path = os.path.join(outdir, "stats.txt")
    with open(stats_path, 'w') as f:
        f.write("image\tmeanY\tcontrast\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]:.6f}\t{r[2]:.6f}\n")
    print("Saved stats to:", stats_path)
    print("Done. Outputs are in:", os.path.abspath(outdir))
    print("You can tune parameters (--key, --gauss-sigma, --bilateral-sigma-*) and re-run to change results.")

if __name__ == "__main__":
    main()
