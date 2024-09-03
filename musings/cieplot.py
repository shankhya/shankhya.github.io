import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

# List to store the clicked points
points = []
num_points = 0  # Number of points to plot

def xyz_to_rgb(xyz):
    M = np.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570]
    ])

    rgb = np.dot(M, xyz)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * (rgb ** (1/2.4)) - 0.055)
    rgb = np.clip(rgb, 0, 1)
    return rgb

def plot_point(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        points.append([x, y])
        plt.plot(x, y, 'o', color='red')  # Plot the point
        plt.text(x, y, f'({x:.4f}, {y:.4f})', fontsize=12, ha='left')  # Annotate the point
        if len(points) == num_points:  # Check if the user has clicked all points
            update_polygon()  # Draw the convex hull
            plt.draw()  # Redraw the plot with the new point and polygon

def update_polygon():
    if len(points) >= 3:  # Need at least 3 points to form a polygon
        hull = ConvexHull(points)
        polygon_points = [points[i] for i in hull.vertices]
        polygon_points.append(polygon_points[0])  # Close the polygon

        # Draw the convex hull polygon
        polygon = Polygon(polygon_points, closed=True, fill=True, edgecolor='blue', alpha=0.3)
        plt.gca().add_patch(polygon)

def cieplot():
    global num_points

    # Ask the user for the number of points
    num_points = int(input("How many points would you like to plot? "))

    # Load the spectral locus data
    locus = np.array([
        [0.175596, 0.005295], [0.172787, 0.004800], [0.170806, 0.005472], [0.170085, 0.005976],
        [0.160343, 0.014496], [0.146958, 0.026643], [0.139149, 0.035211], [0.133536, 0.042704],
        [0.126688, 0.053441], [0.115830, 0.073601], [0.109616, 0.086866], [0.099146, 0.112037],
        [0.091310, 0.132737], [0.078130, 0.170464], [0.068717, 0.200773], [0.054675, 0.254155],
        [0.040763, 0.317049], [0.027497, 0.387997], [0.016270, 0.463035], [0.008169, 0.538504],
        [0.004876, 0.587196], [0.003983, 0.610526], [0.003859, 0.654897], [0.004646, 0.675970],
        [0.007988, 0.715407], [0.013870, 0.750246], [0.022244, 0.779682], [0.027273, 0.792153],
        [0.032820, 0.802971], [0.038851, 0.812059], [0.045327, 0.819430], [0.052175, 0.825200],
        [0.059323, 0.829460], [0.066713, 0.832306], [0.074299, 0.833833], [0.089937, 0.833316],
        [0.114155, 0.826231], [0.138695, 0.814796], [0.154714, 0.805884], [0.192865, 0.781648],
        [0.229607, 0.754347], [0.265760, 0.724342], [0.301588, 0.692326], [0.337346, 0.658867],
        [0.373083, 0.624470], [0.408717, 0.589626], [0.444043, 0.554734], [0.478755, 0.520222],
        [0.512467, 0.486611], [0.544767, 0.454454], [0.575132, 0.424252], [0.602914, 0.396516],
        [0.627018, 0.372510], [0.648215, 0.351413], [0.665746, 0.334028], [0.680061, 0.319765],
        [0.691487, 0.308359], [0.700589, 0.299317], [0.707901, 0.292044], [0.714015, 0.285945],
        [0.719017, 0.280951], [0.723016, 0.276964], [0.734674, 0.265326]
    ])
    
    # Plot the spectral locus
    plt.plot(locus[:, 0], locus[:, 1], 'k', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('CIE 1931 Chromaticity Diagram')
    
    # Plot the non-spectral locus
    plt.plot([locus[0, 0], locus[-1, 0]], [locus[0, 1], locus[-1, 1]], 'k', linewidth=2)
    
    e = 1/3
    steps = 500  # Increased steps for smoother color transitions
    
    for i in range(len(locus)):
        w2 = (i + 1) % len(locus)
        a1 = np.arctan2(locus[i, 1] - e, locus[i, 0] - e)
        a2 = np.arctan2(locus[w2, 1] - e, locus[w2, 0] - e)
        r1 = np.sqrt((locus[i, 0] - e) ** 2 + (locus[i, 1] - e) ** 2)
        r2 = np.sqrt((locus[w2, 0] - e) ** 2 + (locus[w2, 1] - e) ** 2)
        
        for c in range(1, steps + 1):
            xyz = np.zeros((4, 3))
            xyz[0, 0] = e + r1 * np.cos(a1) * c / steps
            xyz[0, 1] = e + r1 * np.sin(a1) * c / steps
            xyz[1, 0] = e + r1 * np.cos(a1) * (c - 1) / steps
            xyz[1, 1] = e + r1 * np.sin(a1) * (c - 1) / steps
            xyz[2, 0] = e + r2 * np.cos(a2) * (c - 1) / steps
            xyz[2, 1] = e + r2 * np.sin(a2) * (c - 1) / steps
            xyz[3, 0] = e + r2 * np.cos(a2) * c / steps
            xyz[3, 1] = e + r2 * np.sin(a2) * c / steps
            
            xyz[:, 2] = 1 - xyz[:, 0] - xyz[:, 1]
            
            rgb = xyz_to_rgb(xyz.T)
            
            polygon = Polygon(xyz[:, :2], facecolor=rgb.mean(axis=1), edgecolor='none')
            plt.gca().add_patch(polygon)
    
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.9)
    plt.grid(False)  # Turn the grid back on
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Connect the plot to a mouse click event
    plt.gcf().canvas.mpl_connect('button_press_event', plot_point)
    plt.show()

cieplot()
