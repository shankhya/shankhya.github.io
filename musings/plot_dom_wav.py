import colour
import matplotlib.pyplot as plt

# Define the points
x1y1 = [0.35, 0.35]
x2y2 = [0.28, 0.56]

# Calculate the dominant wavelengths and other parameters
wl1, xi1, xii1 = colour.dominant_wavelength(x1y1, [1 / 3, 1 / 3])
wl2, xi2, xii2 = colour.dominant_wavelength(x2y2, [1 / 3, 1 / 3])

# Plot the CIE 1931 Chromaticity Diagram
colour.plotting.plot_chromaticity_diagram_CIE1931(show=False)

# Plot the points and lines
plt.plot([1 / 3, x1y1[0], xi1[0]], [1 / 3, x1y1[1], xi1[1]], "o-")
plt.plot([1 / 3, x2y2[0], xi2[0]], [1 / 3, x2y2[1], xi2[1]], "o-")

# Annotate the points with text
plt.annotate('[0.35, 0.35]', xy=(x1y1[0], x1y1[1]), xytext=(x1y1[0] + 0.02, x1y1[1] - 0.02))
plt.annotate(f'Dom Wavelength: {wl1:.2f} nm', xy=(xi1[0], xi1[1]), xytext=(xi1[0] + 0.02, xi1[1] + 0.02))

plt.annotate('[0.28, 0.56]', xy=(x2y2[0], x2y2[1]), xytext=(x2y2[0] + 0.02, x2y2[1] - 0.02))
plt.annotate(f'Dom Wavelength: {wl2:.2f} nm', xy=(xi2[0], xi2[1]), xytext=(xi2[0] + 0.02, xi2[1] - 0.02))
plt.annotate('White point', xy=(1/3, 1/3), xytext=(1/3  -0.02, 1/3 - 0.04))


# Save the figure
plt.savefig('dom.png')