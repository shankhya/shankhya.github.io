import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
def plot_lab_gamut_2views(image_path, theta_segments=12, phi_segments=6):
    # --- Read and convert image ---
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    # --- Convert LAB channels to approximate CIE ranges ---
    L_chan = lab_image[:, :, 0].astype(np.float32) * (100.0 / 255.0)  # [0..100]
    a_chan = lab_image[:, :, 1].astype(np.float32) - 128.0           # [-128..127]
    b_chan = lab_image[:, :, 2].astype(np.float32) - 128.0           # [-128..127]
    # --- Create (L, a, b) array of all pixels ---
    lab_points = np.column_stack((L_chan.flatten(), a_chan.flatten(), b_chan.flatten()))
    # (Optional) Filter out near-black points
    mask = np.linalg.norm(lab_points, axis=1) > 10
    lab_points = lab_points[mask]
    colors = rgb_image.reshape(-1, 3)[mask] / 255.0
    # --- Convert (L,a,b) to spherical coords for segment maxima ---
    x = lab_points[:, 1]  # a*
    y = lab_points[:, 2]  # b*
    z = lab_points[:, 0]  # L*
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-8
    theta = np.arctan2(y, x)   # [-π, π]
    phi = np.arccos(z / r)     # [0, π]
    # --- Create angular bins and find maxima points in each segment ---
    theta_bins = np.linspace(-np.pi, np.pi, theta_segments + 1)
    phi_bins = np.linspace(0, np.pi, phi_segments + 1)
    maxima_points = []
    for i in range(theta_segments):
        for j in range(phi_segments):
            theta_min, theta_max = theta_bins[i], theta_bins[i + 1]
            phi_min, phi_max = phi_bins[j], phi_bins[j + 1]
            seg_mask = (theta >= theta_min) & (theta < theta_max) & \
                       (phi   >= phi_min)   & (phi   < phi_max)
            if np.any(seg_mask):
                seg_pts = lab_points[seg_mask]
                seg_r   = r[seg_mask]
                # pick the point with largest radius in that segment
                maxima_points.append(seg_pts[np.argmax(seg_r)])
    if not maxima_points:
        print("No maxima points found!")
        return
    maxima_points = np.array(maxima_points)
    # --- Build a convex hull on the maxima points ---
    try:
        hull = ConvexHull(maxima_points)
    except:
        print("Not enough points for a convex hull.")
        return
    # =============== PREPARE FIGURE AND SUBPLOTS ===============
    fig = plt.figure(figsize=(16, 7))
    # Left subplot: unfilled wireframe
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlabel('a*')
    ax1.set_ylabel('b*')
    ax1.set_zlabel('L*')
    ax1.set_title('Gamut Boundary - Wireframe')
    # Right subplot: color-filled faces
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlabel('a*')
    ax2.set_ylabel('b*')
    ax2.set_zlabel('L*')
    ax2.set_title('Gamut Boundary - Filled Faces')
    # =============== PLOT POINTS IN BOTH SUBPLOTS ===============
    # Recalling that for plotting, we use (a, b, L)
    ax1.scatter(x, y, z, c=colors, marker='.', alpha=0.3, s=10)
    ax2.scatter(x, y, z, c=colors, marker='.', alpha=0.3, s=10)
    # =============== LEFT SUBPLOT: WIREFRAME ===============
    # Draw lines for each triangle in the hull
    for simplex in hull.simplices:
        pts = maxima_points[simplex]  # shape (3,3) => (L,a,b)
        # reorder (L,a,b)->(a,b,L) for plotting
        tri = pts[:, [1, 2, 0]]
        # We'll connect each edge in tri:
        for i in range(3):
            start = tri[i]
            end   = tri[(i+1)%3]
            ax1.plot([start[0], end[0]],
                     [start[1], end[1]],
                     [start[2], end[2]],
                     color='darkred', linewidth=1.0)
    # =============== RIGHT SUBPLOT: FILLED FACES ===============
    triangles = []
    face_colors = []
    for simplex in hull.simplices:
        # simplex is an index triplet for a triangle
        tri_coords_lab = maxima_points[simplex]  # shape (3, 3) => (L,a,b)
        # Reorder each vertex from (L,a,b) -> (a,b,L) for plotting in 3D
        tri_coords_plot = tri_coords_lab[:, [1, 2, 0]].tolist()
        triangles.append(tri_coords_plot)
        # Compute average LAB for the face
        avg_lab = np.mean(tri_coords_lab, axis=0)  # (L,a,b)
        # Convert that average from (L,a,b) in [0..100, -128..127, -128..127]
        # to a Lab [0..255] image pixel that OpenCV can convert to BGR
        L_255 = np.clip(avg_lab[0] * (255.0 / 100.0), 0, 255)
        a_255 = np.clip(avg_lab[1] + 128.0, 0, 255)
        b_255 = np.clip(avg_lab[2] + 128.0, 0, 255)
        lab_pixel_255 = np.array([[[L_255, a_255, b_255]]], dtype=np.uint8)
        # Convert Lab -> BGR -> RGB
        bgr_pixel = cv2.cvtColor(lab_pixel_255, cv2.COLOR_Lab2BGR)
        rgb_pixel = bgr_pixel[0, 0, ::-1] / 255.0  # scale to [0..1]
        face_colors.append(rgb_pixel)
    mesh = Poly3DCollection(triangles, facecolors=face_colors, edgecolors='k', alpha=0.2)
    ax2.add_collection3d(mesh)
    plt.tight_layout()
    plt.show()
# Example usage:
if __name__ == "__main__":
    plot_lab_gamut_2views('sample.jpg', theta_segments=150, phi_segments=150) 
