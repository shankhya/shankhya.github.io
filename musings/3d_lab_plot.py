import plotly.graph_objs as go
import plotly.offline as pyo
import pandas as pd
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

# Read LAB values from an Excel file
lab_df = pd.read_excel('lab_data_plot.xlsx')  # Replace 'lab_data_plot.xlsx' with your Excel file name
L_values = lab_df['L*']
a_values = lab_df['a*']
b_values = lab_df['b*']

# Convert all LAB colors to RGB colors and format them
rgb_colors = []
for L, a, b in zip(L_values, a_values, b_values):
    lab_color = LabColor(L, a, b)
    rgb_color = convert_color(lab_color, sRGBColor)
    rgb_color = [rgb_color.rgb_r * 255, rgb_color.rgb_g * 255, rgb_color.rgb_b * 255]
    rgb_colors.append('rgb({:.0f}, {:.0f}, {:.0f})'.format(*rgb_color))

# Create hover text with LAB notation for each point
hover_text = ['L*: {:.2f}<br>a*: {:.2f}<br>b*: {:.2f}'.format(L, a, b) for L, a, b in zip(L_values, a_values, b_values)]

# Create scatter plot for each LAB point
scatter_points = go.Scatter3d(
    x=a_values,
    y=b_values,
    z=L_values,
    mode='markers',
    marker=dict(
        size=5,
        color=rgb_colors,  # RGB color based on LAB values
    ),
    text=hover_text,  # Assign hover text
    hoverinfo='text'  # Set hover information to display text
)

# Create axis lines without hover text for better visualization
axis_lines = [
    go.Scatter3d(
        x=[0, 128],
        y=[0, 0],
        z=[50, 50],
        mode='lines',
        line=dict(color='black', width=5),
        showlegend=False
    ),
    go.Scatter3d(
        x=[-128, 0],
        y=[0, 0],
        z=[50, 50],
        mode='lines',
        line=dict(color='black', width=5),
        showlegend=False
    ),
    go.Scatter3d(
        x=[0, 0],
        y=[-128, 0],
        z=[50, 50],
        mode='lines',
        line=dict(color='black', width=5),
        showlegend=False
    ),
    go.Scatter3d(
        x=[0, 0],
        y=[0, 128],
        z=[50, 50],
        mode='lines',
        line=dict(color='black', width=5),
        showlegend=False
    ),
    go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, 100],
        mode='lines',
        line=dict(color='black', width=5),
        showlegend=False
    )
]

layout = go.Layout(
    title='Visualization of the FOGRA39 characterization dataset',
    scene=dict(
        xaxis=dict(title='a*', range=[-128, 127], showgrid=False, zeroline=False, showbackground=False,
                   ticks='', showticklabels=False),  # Remove ticks and tick labels
        yaxis=dict(title='b*', range=[-128, 127], showgrid=False, zeroline=False, showbackground=False,
                   ticks='', showticklabels=False),  # Remove ticks and tick labels
        zaxis=dict(title='L*', range=[0, 100], showgrid=False, zeroline=False, showbackground=False,
                   ticks='', showticklabels=False),  # Remove ticks and tick labels
        xaxis_showspikes=False,
        yaxis_showspikes=False,
        zaxis_showspikes=False,
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(
            eye=dict(x=1.25, y=1.25, z=1.25)  # Adjust camera to look at the plot from an isometric angle
        )
    ),
    scene_aspectmode='cube',
    annotations=[
        dict(
            text='Shankhya Debnath, 2024',
            x=1,
            y=0,
            xref='paper',
            yref='paper',
            xanchor='right',
            yanchor='bottom',
            showarrow=False,
        )
    ]
)

data = [scatter_points] + axis_lines

fig = go.Figure(data=data, layout=layout)

# Save the plot as an HTML file
pyo.plot(fig, filename='3d_scatter_lab.html')
