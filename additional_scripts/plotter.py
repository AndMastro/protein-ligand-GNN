import plotly.express as px
import plotly.io as pio

import pandas as pd





import plotly.express as px
import pandas as pd
import plotly.io as pio

import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio

# Sample data
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio






##### grouped stacked bar charts

affinity_levels = ['Low', 'Medium', 'High']
gnn_models = ['GCN', 'GAT', 'GIN', "GINE", "GraphSAGE", "GC-GNN"]
edge_types = ["Protein", "Ligand", "Interaction"]

# Data for Group 1
gnn1_edges_affinity = [
    [4.8, 75.0, 20.2], #low affinty. each value is for protein, ligand, interaction
    [8.8, 64.2, 27],
    [7,70.6,22.4]      # Subgroup A, B, C values for Category 1, 2, 3
]
gnn2_edges_affinity = [
    [10.4, 72.8, 16.8],
    [11.2, 60.6, 28.2],
    [13.0,63.4,23.6]      # Subgroup A, B, C values for Category 1, 2, 3
]

gnn3_edges_affinity = [
    [10.8,72.4,16.8],
    [14.8,58.4,26.8],
    [22.4,47.6,30.0]      # Subgroup A, B, C values for Category 1, 2, 3
]

gnn4_edges_affinity = [
    [6.7,65.3,28.0],
    [11.8,59.4,28.8],
    [10.6,62.2,27.2]      # Subgroup A, B, C values for Category 1, 2, 3
]

gnn5_edges_affinity = [
    [10.3,70.2,19.5],
    [9.8,67.6,22.6],
    [10.2,69.8,20.0]      # Subgroup A, B, C values for Category 1, 2, 3
]

gnn6_edges_affinity = [
    [11.1,55.2,33.7],
    [12.0,42.6,45.4],
    [11.8,36.6,51.6]      # Subgroup A, B, C values for Category 1, 2, 3
]

subgroup_colors = {
    'Protein': 'darkblue',   # 
    'Ligand': 'red',  # 
    'Interaction': 'green'   # 
}
# Combine all data into a single DataFrame
data = []
for i, affinity in enumerate(affinity_levels):
    for j, gnn_model in enumerate(gnn_models):
        for k, edge_type in enumerate(edge_types):
            group_data = eval(f'gnn{j+1}_edges_affinity')
            # print(group_data[i][k])
            # # sys.exit()
            data.append([affinity, gnn_model, edge_type, group_data[i][k]])

df = pd.DataFrame(data, columns=['Affinity', 'GNN Model', 'Edge Type', '% top-25 edges'])
# print(df)
group_labels_custom = {
    'GCN': 'GCN',
    'GAT': 'GAT',
    'GIN': 'GIN',
    'GINE': 'GINE',
    'GraphSAGE': 'GraphSAGE',
    'GC-GNN': 'GC-GNN'
}


# Create the grouped stacked bar chart with multiple subgroups
fig = px.bar(df, x='Affinity', y='% top-25 edges', color='Edge Type',
             barmode='stack', facet_col='GNN Model',
             category_orders={'Categories': affinity_levels},
             color_discrete_map=subgroup_colors,
             labels={'Affinity': 'Affinity', 'GNN Model': '', 'Edge Type': ''},
             title='',
             text=df['% top-25 edges'])

# fig.update_layout(xaxis_tickangle=45)


width_px = 1000
height_px = int(width_px * 9 / 16) # 16:9 aspect ratio


pio.write_image(fig, "plots/important_edges_edgeshaper.png", format='png', width=width_px, height=height_px, scale=3)#, scale=3)


# Add annotations for the values in the middle of each stack
# for trace in fig['data']:
#     y_offset = [0] * len(trace['x'])  # Initialize y_offset for each group
#     for i in range(len(trace['x'])):
#         x = trace['x'][i]
#         y = trace['y'][i]
#         if y < 0:
#             y_offset[i] += y  # Adjust for negative values
#         annotation_text = str(y)  # Convert the value to string for annotation
#         fig.add_annotation(x=x, y=(y + y_offset[i]), text=annotation_text,
#                            showarrow=False, font=dict(size=10),
#                            xanchor='center', yanchor='bottom')
#         if y >= 0:
#             y_offset[i] += y  # Adjust for positive values
        
# fig.show()