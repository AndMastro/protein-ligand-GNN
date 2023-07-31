import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import plotly.graph_objects as go
import pandas as pd

# pio.renderers.default = "png"
# Sample data
# categories = ['GCN', 'GAT', 'GIN', "GINE", "GraphSAGE", "GC-GNN"]
# low = [20, 30, 25, 10, 15, 10]
# medium = [15, 20, 35, 25, 20, 25]
# high = [10, 25, 20, 30, 25, 35]

# fig = go.Figure()

# # Add bars for each group
# fig.add_trace(go.Bar(x=categories, y=low, name='Low Affinity'))
# fig.add_trace(go.Bar(x=categories, y=medium, name='Medium Affinity'))
# fig.add_trace(go.Bar(x=categories, y=high, name='High Affinity'))

# # Update the layout
# fig.update_layout(
#                   xaxis_title='GNN model',
#                   yaxis_title='% top-25 edges',
#                   barmode='stack')  # Use 'group' for grouped bars



# filename = 'important_edges_edgeshaper'
# extension = 'png'



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
    [10.2, 71.2, 16.4],
    [11.2, 60.6, 28.2],
    [13.0,63.4,23.6]      # Subgroup A, B, C values for Category 1, 2, 3
]

gnn3_edges_affinity = [
    [10.6,71.4,16.6],
    [14.8,58.4,26.8],
    [22.4,47.6,30.0]      # Subgroup A, B, C values for Category 1, 2, 3
]

gnn4_edges_affinity = [
    [6.6,64.4,27.6],
    [11.8,59.4,28.8],
    [10.6,62.2,27.2]      # Subgroup A, B, C values for Category 1, 2, 3
]

gnn5_edges_affinity = [
    [10.2,69.2,19.2],
    [9.8,67.6,22.6],
    [10.2,69.8,20.0]      # Subgroup A, B, C values for Category 1, 2, 3
]

gnn6_edges_affinity = [
    [11.0,54.4,33.2],
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
# Create the grouped stacked bar chart with multiple subgroups
fig = px.bar(df, x='Affinity', y='% top-25 edges', color='Edge Type',
             barmode='stack', facet_col='GNN Model',
             category_orders={'Categories': affinity_levels},
             color_discrete_map=subgroup_colors,
             labels={'Affinity': '', 'GNN Model': '', 'Edge Type': ''},
             title='')

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
pio.write_image(fig, "important_edges_edgeshaper.png", scale=3)#, scale=3)