# Attempt to implement a CIECAM02 demo.
# This notebook-style script will:
# 1. Try to use the `colour` library for an accurate CIECAM02 implementation.
# 2. If `colour` isn't available, fall back to a simplified approximation using CIELAB and clearly notify the user.
# 3. Generate a palette of colors (varying hue and lightness), compute appearance attributes under three viewing conditions,
#    and plot the results (J (lightness), C (chroma), and h (hue angle) for each condition).
# 4. Display the palette images under different viewing conditions (via the appearance model when available).
#
# The code will create visual output within this notebook environment.
import sys
import numpy as np
import matplotlib.pyplot as plt

# Helper: generate a palette of colors in sRGB by sweeping hue and lightness
def generate_palette(num_hues=12, num_lightness=5):
    # We'll generate colors in HCL-like space via converting from CIELAB-ish approximation.
    hues = np.linspace(0, 360, num_hues, endpoint=False)
    lights = np.linspace(30, 90, num_lightness)  # perceptual lightness-like
    palette = []
    coords = []
    for L in lights:
        row = []
        row_coords = []
        for h in hues:
            # Use HCL -> approximate to sRGB via simple conversion:
            # We'll create an sRGB color from H,S,V like HSV but tuned to look good.
            # Use H (hue), S (fixed), V from L normalized.
            H = h / 360.0
            S = 0.75
            V = L / 100.0
            # Convert HSV -> RGB
            i = int(H*6)
            f = (H*6) - i
            p = V * (1 - S)
            q = V * (1 - f*S)
            t = V * (1 - (1-f)*S)
            i = i % 6
            if i == 0:
                r,g,b = V,t,p
            elif i == 1:
                r,g,b = q,V,p
            elif i == 2:
                r,g,b = p,V,t
            elif i == 3:
                r,g,b = p,q,V
            elif i == 4:
                r,g,b = t,p,V
            else:
                r,g,b = V,p,q
            row.append(np.clip([r,g,b], 0, 1))
            row_coords.append((L, h))
        palette.append(row)
        coords.append(row_coords)
    palette = np.array(palette)  # shape (num_lightness, num_hues, 3)
    return palette, coords

palette, coords = generate_palette(num_hues=24, num_lightness=6)

# Flatten colors for processing
colors_rgb = palette.reshape(-1, 3)

# We'll try to use 'colour' for true CIECAM02. If it isn't available, we'll fall back to CIELAB-based approx.
use_colour = False
try:
    import colour
    from colour.appearance import CIECAM02_Specification, CIECAM02_InductionFactors, XYZ_to_CIECAM02, CIECAM02_to_XYZ
    # Also import colour.models for sRGB <-> XYZ
    from colour import sRGB_to_XYZ, XYZ_to_sRGB, xyY_to_XYZ, XYZ_to_xyY
    use_colour = True
except Exception as e:
    print("`colour` library not available in this environment. Falling back to a CIELAB-based approximation for demo purposes.")
    print("If you want a full CIECAM02 implementation, please install the 'colour-science' package and re-run.")
    # We'll use skimage's rgb2lab if available, else implement a simple srgb->lab via colorspacious or none.
    try:
        from skimage import color as skcolor
        has_skimage = True
    except Exception:
        has_skimage = False

# Define three viewing conditions (Average, Dim, Dark) - values taken as typical examples
viewing_conditions = {
    "Average": {"L_A": 64, "Y_b": 20, "surround": "average"},
    "Dim": {"L_A": 20, "Y_b": 20, "surround": "dim"},
    "Dark": {"L_A": 5, "Y_b": 20, "surround": "dark"}
}

# Define a helper to compute CIECAM02 attributes using colour if available
def compute_ciecam02_with_colour(rgb_colors, condition_name):
    # rgb_colors: Nx3 in linear sRGB (0-1)
    # Convert to XYZ (assume sRGB in D65)
    xyz = sRGB_to_XYZ(rgb_colors)
    # Build viewing condition objects (approximate)
    if condition_name == "Average":
        F = 1.0; c = 0.69; N_c = 1.0
        L_A = 64; Y_b = 20.0
    elif condition_name == "Dim":
        F = 0.9; c = 0.59; N_c = 0.95
        L_A = 20; Y_b = 20.0
    else:
        F = 0.8; c = 0.525; N_c = 0.8
        L_A = 5; Y_b = 20.0
    # whitepoint D65 XY Z:
    xyY = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    # convert to XYZ reference white (Y=100)
    XYZ_w = xyY_to_XYZ((xyY[0], xyY[1], 100.0))
    # Induction factors object
    induction = CIECAM02_InductionFactors(F=F, c=c, N_c=N_c)
    # Compute CAM02 for each sample
    specs = []
    for X in xyz:
        spec = XYZ_to_CIECAM02(X, XYZ_w, L_A, Y_b, induction)
        specs.append(spec)
    # specs is list of CIECAM02_Specification objects; extract J,C,h
    J = np.array([s.J for s in specs])
    C = np.array([s.C for s in specs])
    h = np.array([s.h for s in specs])
    return J, C, h

# Fallback: compute approximate "appearance attributes" using CIELAB if `colour` is not present
def compute_approx_appearance_with_lab(rgb_colors, condition_name):
    # Use skimage if available to compute Lab (D65)
    if has_skimage:
        lab = skcolor.rgb2lab(rgb_colors.reshape(1,-1,3)).reshape(-1,3)
    else:
        # simple approximate linear -> xyz -> lab conversion (not colorimetrically precise)
        def srgb_to_linear(rgb):
            rgb = np.clip(rgb, 0, 1)
            mask = rgb <= 0.04045
            lin = np.where(mask, rgb/12.92, ((rgb+0.055)/1.055)**2.4)
            return lin
        def linear_srgb_to_xyz(rgb_lin):
            M = np.array([[0.4124564,0.3575761,0.1804375],
                          [0.2126729,0.7151522,0.0721750],
                          [0.0193339,0.1191920,0.9503041]])
            return rgb_lin @ M.T
        def xyz_to_lab(xyz):
            # reference white D65 (Y=100)
            Xn, Yn, Zn = 95.047, 100.0, 108.883
            xyz_scaled = np.stack([xyz[:,0]/Xn, xyz[:,1]/Yn, xyz[:,2]/Zn], axis=1)
            def f(t):
                delta = 6/29
                return np.where(t > delta**3, t**(1/3), (t/(3*delta**2)) + 4/29)
            fxyz = f(xyz_scaled)
            L = 116 * fxyz[:,1] - 16
            a = 500 * (fxyz[:,0] - fxyz[:,1])
            b = 200 * (fxyz[:,1] - fxyz[:,2])
            return np.stack([L,a,b], axis=1)
        rgb_lin = srgb_to_linear(rgb_colors)
        xyz = linear_srgb_to_xyz(rgb_lin) * 100.0  # scale to 0-100
        lab = xyz_to_lab(xyz)
    # Now create approximate J (use L), C (sqrt(a^2+b^2)), h (atan2(b,a) in degrees)
    L = lab[:,0]
    a = lab[:,1]; b = lab[:,2]
    J = L.copy()
    C = np.sqrt(a*a + b*b)
    h = (np.degrees(np.arctan2(b, a)) + 360) % 360
    # Simulate slight shift by viewing condition (just for demo): scale J and C for Dim/Dark
    if condition_name == "Dim":
        J = J * 0.9
        C = C * 0.95
    elif condition_name == "Dark":
        J = J * 0.75
        C = C * 0.9
    return J, C, h

# Compute appearance attributes for each viewing condition
results = {}
if use_colour:
    for name in viewing_conditions:
        J, C, h = compute_ciecam02_with_colour(colors_rgb, name)
        results[name] = {"J":J, "C":C, "h":h}
else:
    for name in viewing_conditions:
        J, C, h = compute_approx_appearance_with_lab(colors_rgb, name)
        results[name] = {"J":J, "C":C, "h":h}

# Visualization: show palette as image and plots of J & C for each condition
num_rows, num_cols, _ = palette.shape
fig, axes = plt.subplots(2, 4, figsize=(16, 8), gridspec_kw={'height_ratios':[3,1]})
# Display original palette
axes[0,0].imshow(palette)
axes[0,0].set_title("Palette (sRGB)")
axes[0,0].axis('off')

# For each viewing condition, show the "appearance" mapped image (if colour available, we would simulate adaptation)
for i, (name, vals) in enumerate(results.items()):
    row = i // 3
    col = i % 3 + 1  # place in axes[0,1..3]
    ax_img = axes[0, col]
    # For demo, we'll just show the same palette but annotate title with condition and sample stats
    ax_img.imshow(palette)
    ax_img.set_title(f"{name} (L_A={viewing_conditions[name]['L_A']})")
    ax_img.axis('off')
    # Plot J and C as heatmaps below
    ax_j = axes[1, col]
    Jmap = vals['J'].reshape(num_rows, num_cols)
    Cmap = vals['C'].reshape(num_rows, num_cols)
    im = ax_j.imshow(Jmap, aspect='auto')
    ax_j.set_title(f"J (lightness) - {name}")
    ax_j.axis('off')
    fig.colorbar(im, ax=ax_j, fraction=0.046, pad=0.01)

# Hide the last unused subplot
axes[1,0].axis('off')
axes[0,3].axis('off')

plt.tight_layout()
plt.show()

# Additionally show scatter plots of J vs C for each condition
plt.figure(figsize=(8,6))
for name, vals in results.items():
    plt.scatter(vals['C'], vals['J'], label=name, alpha=0.7)
plt.xlabel("C (chroma)")
plt.ylabel("J (lightness)")
plt.title("J vs C for palette under different viewing conditions")
plt.legend()
plt.grid(True)
plt.show()

# Display a small table of first 12 sample colors and their J,C,h under each condition
import pandas as pd
N = min(12, len(colors_rgb))
rows = []
for i in range(N):
    r,g,b = colors_rgb[i]
    row = {"sR":round(r,3), "sG":round(g,3), "sB":round(b,3)}
    for name, vals in results.items():
        row[f"J_{name}"] = round(float(vals['J'][i]),2)
        row[f"C_{name}"] = round(float(vals['C'][i]),2)
        row[f"h_{name}"] = round(float(vals['h'][i]),1)
    rows.append(row)
df = pd.DataFrame(rows)
import caas_jupyter_tools as cjt
cjt.display_dataframe_to_user("CIECAM02 Demo Samples", df)

# Save a small report image
out_path = "/mnt/data/ciecam02_demo_palette.png"
fig.savefig(out_path)
print(f"Saved a summary figure to: {out_path}")
