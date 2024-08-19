import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the Excel file into a DataFrame
df = pd.read_excel('references.xlsx')

# Exclude the "OTHERS" category
df = df[df['Category'] != 'OTHERS']

# Rename the categories
category_mapping = {
    'COL': 'Color Management',
    'INV': 'Inverse Halftoning',
    'HAL': 'Digital Halftoning',
    'OPT': 'Process Optimization',
    'PRO': 'Process Control',
    'QUA': 'Quality Control',
    'FOR': 'Printer Forensics'
}

df['Category'] = df['Category'].map(category_mapping)

# Calculate the total number of papers in each area
papers_per_area = df['Category'].value_counts()

# Calculate the total number of papers in the plot
total_papers = df.shape[0]

# Group the data by 'Year' and 'Category' and count the number of papers in each group
grouped_df = df.groupby(['Year of Publication', 'Category']).size().reset_index(name='Count')

# Join titles of papers with the grouped DataFrame based on 'Year' and 'Category'
grouped_df = pd.merge(grouped_df, df[['Year of Publication', 'Category', 'Title of Publication']], on=['Year of Publication', 'Category'], how='left')

# Create hover text to display all paper names for each scatter point
hover_text = grouped_df.groupby(['Year of Publication', 'Category'])['Title of Publication'].apply(lambda x: '<br>'.join(x)).reset_index(name='HoverText')

# Merge hover text with grouped data
grouped_df = pd.merge(grouped_df, hover_text, on=['Year of Publication', 'Category'], how='left')

# Create a scatter plot with interactivity
fig = px.scatter(grouped_df, x='Year of Publication', y='Category', color='Category', size='Count',
                 hover_name='Category', hover_data={'Year of Publication': True, 'Count': True, 'HoverText': True},
                 title="Use of computational intelligence in research on printing technology",
                 labels={'Year of Publication': 'Year of Publication', 'Category': 'Category', 'Count': 'Number of Papers'},
                 size_max=20)  # Adjust the maximum size of scatter points as needed

# Add annotations for the number of papers in each area
annotations = []
for area, count in papers_per_area.items():
    annotations.append(
        dict(
            xref="paper",
            yref="y",
            x=1.08,  # Move the annotation more to the left
            y=area,
            text=f"{count} Papers",
            xanchor='right',  # Anchor to the right side of the annotation
            showarrow=False,
            font=dict(size=10),
        )
    )

# Add annotation for total number of papers in the plot
annotations.append(
    dict(
        xref="paper", yref="paper",
        x=1, y=-0.1,
        text=f"Total Papers in Plot: {total_papers}",
        showarrow=False,
        font=dict(size=10),
    )
)

# Add the annotations to the plot
fig.update_layout(annotations=annotations)

# Customize the legend position
fig.update_layout(
    legend=dict(
        x=1.11,  # Move the legend more to the right
        y=1
    )
)

# Add annotation for your name
fig.add_annotation(
    text="Shankhya Debnath",
    xref="paper", yref="paper",
    x=1, y=0,
    showarrow=False,
    xanchor='right', yanchor='bottom',
    font=dict(size=12, color="black")
)

# Save the plot as an HTML file
fig.write_html("plot1.html")
