import matplotlib.pyplot as plt
from colour.plotting import plot_chromaticity_diagram_CIE1931
import numpy as np
from matplotlib.patches import Polygon

c25=[0.299058325,	0.307562399]
c50=[0.285112598,	0.292562491]
c75=[0.260675523,	0.270196374]
c100=[0.246194172,	0.24496245]
m25=[0.321834155,	0.311833023]
m50=[0.341566408,	0.313036017]
m75=[0.372587353,	0.313455615]
m100=[0.38110501,	0.312550367]
y25=[0.324367074,	0.334329843]
y50=[0.340177339,	0.356231777]
y75=[0.381015223,	0.408634471]
y100=[0.380449312,	0.394958341]
cm25=[0.30385182,	0.300144556]
cm50=[0.295735604,	0.27917404]
cm75=[0.282282727,	0.254243024]
cm100=[0.289480313,	0.271106483]
my25=[0.337006678,	0.331722268]
my50=[0.368894465,	0.346705788]
my75=[0.403744413,	0.347493867]
my100=[0.389523115,	0.329678358]
cy25=[0.307763816,	0.323157716]
cy50=[0.302134921,	0.331829834]
cy75=[0.287515204,	0.350695665]
cy100=[0.280955932,	0.352167336]
d65=[0.312720521,	0.329030684]

plot_chromaticity_diagram_CIE1931(standalone=False)
plt.plot(d65[0], d65[1], ".", color=(0, 0, 0))
plt.plot(c25[0], c25[1], ".", color=(0, 0.7, 0.7))
#plt.annotate('Cyan', xy=(c25[0]+0.003, c25[1]+0.003))
plt.plot(c50[0], c50[1], ".", color=(0, 0.8, 0.8))
plt.plot(c75[0], c75[1], ".", color=(0, 0.9, 0.9))
plt.plot(c100[0], c100[1], ".", color=(0, 1, 1))
plt.plot(m25[0], m25[1], ".", color=(0.7, 0, 0.7))
plt.plot(m50[0], m50[1], ".", color=(0.8, 0, 0.8))
plt.plot(m75[0], m75[1], ".", color=(0.9, 0, 0.9))
plt.plot(m100[0], m100[1], ".", color=(1, 0, 1))
plt.plot(y25[0], y25[1], ".", color=(0.7, 0.7, 0))
plt.plot(y50[0], y50[1], ".", color=(0.8, 0.8, 0))
plt.plot(y75[0], y75[1], ".", color=(0.9, 0.9, 0))
plt.plot(y100[0], y100[1], ".", color=(1, 1, 0))
plt.plot(cm25[0], cm25[1], ".", color=(0, 0, 0.7))
plt.plot(cm50[0], cm50[1], ".", color=(0, 0, 0.8))
plt.plot(cm75[0], cm75[1], ".", color=(0, 0, 0.9))
plt.plot(cm100[0], cm100[1], ".", color=(0, 0, 1))
plt.plot(my25[0], my25[1], ".", color=(0.7, 0, 0))
plt.plot(my50[0], my50[1], ".", color=(0.8, 0, 0))
plt.plot(my75[0], my75[1], ".", color=(0.9, 0, 0))
plt.plot(my100[0], my100[1], ".", color=(1, 0, 0))
plt.plot(cy25[0], cy25[1], ".", color=(0, 0.7, 0))
plt.plot(cy50[0], cy50[1], ".", color=(0, 0.8, 0))
plt.plot(cy75[0], cy75[1], ".", color=(0, 0.9, 0))
plt.plot(cy100[0], cy100[1], ".", color=(0, 1, 0))
points=[(0.280955932,	0.352167336), (0.246194172,	0.24496245), (0.282282727,	0.254243024), (0.38110501,	0.312550367), (0.403744413,	0.347493867),(0.381015223,	0.408634471) ]
x_coords, y_coords = zip(*points)
boundary = Polygon(points, closed=True, edgecolor='black', fill=False)
plt.scatter(x_coords, y_coords, color='blue', marker='.', label='Points')
plt.gca().add_patch(boundary)

degree = 3
cx=[d65[0], c25[0], c50[0], c75[0], c100[0]]
cy=[d65[1], c25[1], c50[1], c75[1], c100[1]]
coefficients = np.polyfit(cx, cy, degree)
cx_curve = np.linspace(min(cx), max(cx), 100)
cy_curve = np.polyval(coefficients, cx_curve)
plt.plot(cx_curve, cy_curve, color=(0, 0.7, 0.7), label='Fitted Curve')

degree = 3
mx=[d65[0], m25[0], m50[0], m75[0], m100[0]]
my=[d65[1], m25[1], m50[1], m75[1], m100[1]]
coefficients = np.polyfit(mx, my, degree)
mx_curve = np.linspace(min(mx), max(mx), 100)
my_curve = np.polyval(coefficients, mx_curve)
plt.plot(mx_curve, my_curve, color=(0.7, 0, 0.7), label='Fitted Curve')

degree = 3
yx=[d65[0], y25[0], y50[0], y75[0], y100[0]]
yy=[d65[1], y25[1], y50[1], y75[1], y100[1]]
coefficients = np.polyfit(yx, yy, degree)
yx_curve = np.linspace(min(yx), max(yx), 100)
yy_curve = np.polyval(coefficients, yx_curve)
plt.plot(yx_curve, yy_curve, color=(0.7, 0.7, 0), label='Fitted Curve')

degree = 3
cmx=[d65[0], cm25[0], cm50[0], cm75[0], cm100[0]]
cmy=[d65[1], cm25[1], cm50[1], cm75[1], cm100[1]]
coefficients = np.polyfit(cmx, cmy, degree)
cmx_curve = np.linspace(min(cmx), max(cmx), 100)
cmy_curve = np.polyval(coefficients, cmx_curve)
plt.plot(cmx_curve, cmy_curve, color=(0, 0, 0.7), label='Fitted Curve')

degree = 3
cyx=[d65[0], cy25[0], cy50[0], cy75[0], cy100[0]]
cyy=[d65[1], cy25[1], cy50[1], cy75[1], cy100[1]]
coefficients = np.polyfit(cyx, cyy, degree)
cyx_curve = np.linspace(min(cyx), max(cyx), 100)
cyy_curve = np.polyval(coefficients, cyx_curve)
plt.plot(cyx_curve, cyy_curve, color=(0, 0.7, 0), label='Fitted Curve')

degree = 3
myx=[d65[0], my25[0], my50[0], my75[0], my100[0]]
myy=[d65[1], my25[1], my50[1], my75[1], my100[1]]
coefficients = np.polyfit(myx, myy, degree)
myx_curve = np.linspace(min(myx), max(myx), 100)
myy_curve = np.polyval(coefficients, myx_curve)
plt.plot(myx_curve, myy_curve, color=(0.7, 0, 0), label='Fitted Curve')


plt.title('Color gamut of process colors printed on uncoated paper')

plt.savefig('myplot1.png')