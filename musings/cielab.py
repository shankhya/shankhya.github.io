import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Global variables to store clicked points
clicked_points = []
num_points = 0


def cielab_plot(L_value):
    # Define the fixed resolution for the plot
    resolution = 500

    # Define the grid for the a* and b* values
    a_values = np.linspace(-60, 60, resolution)
    b_values = np.linspace(-60, 60, resolution)
    a, b = np.meshgrid(a_values, b_values)

    # Calculate the distance from the origin to each point
    distance = np.sqrt(a ** 2 + b ** 2)

    # Mask values outside the CIELAB color wheel (distance > 50)
    mask = distance <= 50

    # Hue angle (theta), where hue varies with the angle around the origin
    theta = np.arctan2(b, a)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)  # Adjust angle to [0, 2*pi]

    # Chroma is just the distance from the origin
    chroma = np.clip(distance / 50, 0, 1)

    # Adjust lightness factor based on L*
    # Apply lightness scaling where L=50 is neutral and L=0 is black, L=100 is very bright
    L_norm = L_value / 100.0
    L_factor = 0.8 * L_norm + 0.65  # Ensure colors are not too dark, even for lower L* values

    # Calculate RGB values based on LCH values (CIELAB approximation)
    r = chroma * np.clip(np.cos(theta - 0), 0, 1) * L_factor
    g = chroma * np.clip(np.cos(theta - 2 * np.pi / 3), 0, 1) * L_factor
    bl = chroma * np.clip(np.cos(theta - 4 * np.pi / 3), 0, 1) * L_factor

    # Combine the color channels into an RGB image
    rgb_image = np.zeros((a.shape[0], a.shape[1], 3))
    rgb_image[..., 0] = r
    rgb_image[..., 1] = g
    rgb_image[..., 2] = bl

    # Apply the mask to remove the outer part
    rgb_image[~mask] = [1, 1, 1]  # Set outer region to white

    # Create the plot
    plt.imshow(rgb_image, extent=(-60, 60, -60, 60), origin='lower')

    # Plot the axes lines at a* = 0 and b* = 0
    plt.plot([0, 0], [-60, 60], color='black', linewidth=1)
    plt.plot([-60, 60], [0, 0], color='black', linewidth=1)

    # Label the axes and add title
    plt.xlabel('a*')
    plt.ylabel('b*')
    plt.title(f'CIELAB Color Representation at L*={L_value}')

    # Set the aspect ratio to be equal to make it a circle
    plt.gca().set_aspect('equal', adjustable='box')

    # Show ticks and add spacing around plot
    plt.xticks(np.arange(-60, 70, 20))
    plt.yticks(np.arange(-60, 70, 20))

    # Connect to event handler for clicks
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)

    plt.show()


def on_click(event):
    global clicked_points, num_points

    if len(clicked_points) < num_points:
        # Only consider clicks inside the plot area
        if event.xdata is not None and event.ydata is not None:
            # Append the clicked point
            clicked_points.append((event.xdata, event.ydata))
            # Annotate the point with its (a*, b*) values
            plt.gca().annotate(f'({event.xdata:.1f}, {event.ydata:.1f})',
                               (event.xdata, event.ydata),
                               textcoords="offset points", xytext=(5, 5),
                               ha='center', fontsize=8, color='black')
            plt.plot(event.xdata, event.ydata, 'ko')  # Mark the clicked point

            # Redraw plot to show the updated annotations
            plt.draw()

        # Once all points are clicked, calculate and plot the convex hull
        if len(clicked_points) == num_points:
            draw_convex_hull()


def draw_convex_hull():
    global clicked_points

    # Convert the list of points into an array
    points = np.array(clicked_points)
    if len(points) >= 3:  # Convex hull requires at least 3 points
        hull = ConvexHull(points)

        # Plot the convex hull
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k--', lw=2)

    # Redraw the plot to display the convex hull
    plt.draw()


# Main function to get user input and plot the CIELAB color space
def main():
    global num_points

    # Get user input for L* value
    L_value = float(input("Enter the L* value (lightness, typically between 0 and 100): "))

    # Get user input for how many points they want to click
    num_points = int(input("Enter the number of points you'd like to plot: "))

    # Call the function to plot the CIELAB color space with the user-defined L* value
    cielab_plot(L_value)


# Run the main function
if __name__ == "__main__":
    main()