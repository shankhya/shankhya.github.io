from typing import List, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
# Constants and helpers
def to_100_scale(value):
    """Convert to the 0-100 scale."""
    return np.asarray(value) * 100
def from_100_scale(value):
    """Convert from the 0-100 scale."""
    return np.asarray(value) / 100
def spow(x, y):
    """Raise `x` to the power of `y`."""
    return np.sign(x) * np.abs(x) ** y
def cart2pol(x, y):
    """Convert Cartesian coordinates to polar coordinates."""
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return phi, rho
# Surround viewing conditions for CIECAM16 and CIECAM02
class ViewingConditions:
    def __init__(self, F: float, c: float, N_c: float):
        self.F = F  # Degree of adaptation
        self.c = c  # Exponential non-linearity
        self.N_c = N_c  # Chromatic induction factor
# Example average viewing conditions
AVERAGE_VIEWING_CONDITIONS = ViewingConditions(1.0, 0.69, 1.0)
# Specification for the CIECAM16 and CIECAM02 models
@dataclass
class CAMSpec:
    """CIECAM16/CIECAM02 Model Specification"""
    J: Union[float, None] = field(default_factory=lambda: None)  # Lightness correlate
    C: Union[float, None] = field(default_factory=lambda: None)  # Chroma correlate
    h: Union[float, None] = field(default_factory=lambda: None)  # Hue angle
    s: Union[float, None] = field(default_factory=lambda: None)  # Saturation
    Q: Union[float, None] = field(default_factory=lambda: None)  # Brightness
    M: Union[float, None] = field(default_factory=lambda: None)  # Colourfulness
    H: Union[float, None] = field(default_factory=lambda: None)  # Hue quadrature

# CIECAM16 method 
def xyz_to_ciecam16(
        XYZ: List[float], XYZ_w: List[float], L_A: float, Y_b: float,
        surround: ViewingConditions = AVERAGE_VIEWING_CONDITIONS,
        discount_illuminant: bool = False
) -> CAM16Spec:
    """
    Convert CIE XYZ tristimulus values to CIECAM16 color appearance model correlates.
    """
    XYZ = to_100_scale(XYZ)
    XYZ_w = to_100_scale(XYZ_w)
    # Decompose white reference
    X_w, Y_w, Z_w = XYZ_w
    # Convert XYZ to sharpened RGB
    RGB_w = np.dot(MATRIX_16, XYZ_w)
    # Degree of adaptation
    D = np.clip(degree_of_adaptation(surround.F, L_A), 0, 1) if not discount_illuminant else 1
    n, F_L, N_bb, N_cb, z = viewing_conditions_dependent_parameters(Y_b, Y_w, L_A)
    D_RGB = D * 100 / RGB_w + 1 - D
    RGB_wc = D_RGB * RGB_w
    # Non-linear compression
    RGB_aw = post_adaptation_compression(RGB_wc, F_L)
    # Achromatic response
    A_w = achromatic_response(RGB_aw, N_bb)
    # Test sample to sharpened RGB
    RGB = np.dot(MATRIX_16, XYZ)
    # Apply degree of adaptation to test sample
    RGB_c = D_RGB * RGB
    # Non-linear response compression for test sample
    RGB_a = post_adaptation_compression(RGB_c, F_L) + 0.1
    # Convert to opponent color dimensions
    a, b = opponent_colour_dimensions_forward(RGB_a)
    # Compute hue angle
    h = hue_angle(a, b)
    # Eccentricity factor
    e_t = eccentricity_factor(h)
    # Achromatic response for the test sample
    A = achromatic_response(RGB_a, N_bb)
    # Lightness correlate
    J = lightness_correlate(A, A_w, surround.c, z)
    # Brightness correlate
    Q = brightness_correlate(surround.c, J, A_w, F_L)
    # Chroma correlate
    C = chroma_correlate(J, n, surround.N_c, N_cb, e_t, a, b, RGB_a)
    # Colourfulness correlate
    M = colourfulness_correlate(C, F_L)
    # Saturation correlate
    s = saturation_correlate(M, Q)
    return CAM16Spec(J, C, h, s, Q, M, None)
# New CIECAM02 method
def xyz_to_ciecam02(xyz: List[float], xyzw: List[float], la: float, yb: float, para: List[float]) -> CAMSpec:
    f, c, Nc = para
    MH = np.array([[0.38971, 0.68898, -0.07868],
                   [-0.22981, 1.18340, 0.04641],
                   [0.0, 0.0, 1.0]])
    M02 = np.array([[0.7328, 0.4296, -0.1624],
                    [-0.7036, 1.6975, 0.0061],
                    [0.0030, 0.0136, 0.9834]])
    Minv = np.array([[1.096124, -0.278869, 0.182745],
                     [0.454369, 0.473533, 0.072098],
                     [-0.009628, -0.005698, 1.015326]])
    k = 1 / (5 * la + 1)
    fl = (k ** 4) * la + 0.1 * ((1 - k ** 4) ** 2) * ((5 * la) ** (1 / 3))
    n = yb / xyzw[1]
    ncb = 0.725 * (1 / n) ** 0.2
    nbb = ncb
    z = 1.48 + np.sqrt(n)
    # Step 1
    rgb = np.dot(M02, xyz)
    rgbw = np.dot(M02, xyzw)
    # Step 2: Degree of adaptation
    D = f * (1 - (1 / 3.6) * np.exp((-la - 42) / 92))
    # Step 3: Chromatic adaptation
    rgbc = np.zeros_like(rgb)
    for i in range(3):
        rgbc[i] = (D * xyzw[1] / rgbw[i] + 1 - D) * rgb[i]
    rgbwc = np.zeros_like(rgbw)
    for i in range(3):
        rgbwc[i] = (D * xyzw[1] / rgbw[i] + 1 - D) * rgbw[i]
    # Step 4
    rgbp = np.dot(MH, np.dot(Minv, rgbc))
    rgbpw = np.dot(MH, np.dot(Minv, rgbwc))
    # Step 5
    rgbpa = np.zeros_like(rgbp)
    for i in range(3):
        rgbpa[i] = (400 * (fl * rgbp[i] / 100) ** 0.42) / (27.13 + (fl * rgbp[i] / 100) ** 0.42) + 0.1
    rgbpwa = np.zeros_like(rgbpw)
    for i in range(3):
        rgbpwa[i] = (400 * (fl * rgbpw[i] / 100) ** 0.42) / (27.13 + (fl * rgbpw[i] / 100) ** 0.42) + 0.1
    # Step 6
    a = rgbpa[0] - 12 * rgbpa[1] / 11 + rgbpa[2] / 11
    b = (rgbpa[0] + rgbpa[1] - 2 * rgbpa[2]) / 9
    # Step 7: Hue angle
    h, _ = cart2pol(a, b)
    h = np.degrees(h)
    h = np.where(h < 0, h + 360, h)
    # Step 9: Achromatic response
    A = (2 * rgbpa[0] + rgbpa[1] + rgbpa[2] / 20 - 0.305) * nbb
    Aw = (2 * rgbpwa[0] + rgbpwa[1] + rgbpwa[2] / 20 - 0.305) * nbb
    # Step 10: Lightness
    J = 100 * (A / Aw) ** (c * z)
    # Step 11: Brightness
    Q = (4 / c) * (J / 100) ** 0.5 * (Aw + 4) * fl ** 0.25
    # Step 12: Chroma
    et = (np.cos(np.radians(h) + 2) + 3.8) / 4
    t = (Nc * ncb * 50000 / 13) * (et * np.sqrt(a ** 2 + b ** 2)) / (rgbpa[0] + rgbpa[1] + 21 * rgbpa[2] / 20)
    C = (t ** 0.9) * (J / 100) ** 0.5 * (1.64 - 0.29 ** n) ** 0.73
    # Step 13: Colourfulness
    M = C * fl ** 0.25
    # Step 14: Saturation
    s = 100 * (M / Q) ** 0.5
    return CAMSpec(J, C, h, s, Q, M, None)
# Main function to ask for user input and calculate CIECAM16 or CIECAM02
def main():
    print("Do you want to use CIECAM16 or CIECAM02? (16/02)")
    method = input().strip()
    print("Enter the test sample's CIE XYZ values (e.g., 19.01, 20.00, 21.78):")
    XYZ = list(map(float, input().split(',')))
    print("Enter the reference white's CIE XYZ values (e.g., 95.05, 100.00, 108.88):")
    XYZ_w = list(map(float, input().split(',')))
    print("Enter the adapting luminance (L_A) in cd/m² (e.g., 318.31):")
    L_A = float(input())
    print("Enter the luminous factor of the background (Y_b) (e.g., 20.0):")
    Y_b = float(input())
    if method == "16":
        print("Do you want to discount the illuminant? (yes/no):")
        discount_illuminant_input = input().strip().lower()
        discount_illuminant = discount_illuminant_input == "yes"
        # Call the CIECAM16 conversion function
        ciecam16_spec = xyz_to_ciecam16(XYZ, XYZ_w, L_A, Y_b, discount_illuminant=discount_illuminant)
        # Print the results
        print(f"CIECAM16 Model Results:")
        print(f"Lightness (J): {ciecam16_spec.J}")
        print(f"Chroma (C): {ciecam16_spec.C}")
        print(f"Hue angle (h): {ciecam16_spec.h}")
        print(f"Saturation (s): {ciecam16_spec.s}")
        print(f"Brightness (Q): {ciecam16_spec.Q}")
        print(f"Colourfulness (M): {ciecam16_spec.M}")
    elif method == "02":
        print("Enter the parameters for CIECAM02 (f, c, Nc) separated by commas (e.g., 1.0, 0.69, 1.0):")
        para = list(map(float, input().split(',')))
        # Call the CIECAM02 conversion function
        ciecam02_spec = xyz_to_ciecam02(XYZ, XYZ_w, L_A, Y_b, para)
        # Print the results
        print(f"CIECAM02 Model Results:")
        print(f"Lightness (J): {ciecam02_spec.J}")
        print(f"Chroma (C): {ciecam02_spec.C}")
        print(f"Hue angle (h): {ciecam02_spec.h}")
        print(f"Saturation (s): {ciecam02_spec.s}")
        print(f"Brightness (Q): {ciecam02_spec.Q}")
        print(f"Colourfulness (M): {ciecam02_spec.M}")
if __name__ == "__main__":
    main()