#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from tqdm import tqdm
import json
import yaml

import torch
from torch_geometric.data import Data

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx


from src.utils import create_edge_index, PLIDataset



with open("parameters.yml") as paramFile:  
        args = yaml.load(paramFile, Loader=yaml.FullLoader)

DATA_PATH = args["top_k_computation"]["DATA_PATH"]
SAVE_FOLDER = args["top_k_computation"]["SAVE_FOLDER"]
PLOT = args["top_k_computation"]["PLOT"]
EXPLANATIONS_FOLDER = args["top_k_computation"]["EXPLANATIONS_FOLDER"]

TOP_K_VALUES = args["top_k_computation"]["TOP_K_VALUES"]


AFFINITY_GROUPS = ["low affinity", "medium affinity", "high affinity"]

#reduced version of the dataset builder to save computation time and memory
def generate_pli_dataset_dict_reduced(data_path):

    directory = os.fsencode(data_path)

    dataset_dict = {}
    dirs = os.listdir(directory)
    for file in tqdm(dirs):
        interaction_name = os.fsdecode(file)

        
        if os.path.isdir(data_path + interaction_name):
            dataset_dict[interaction_name] = {}
            G = None
            with open(data_path + interaction_name + "/" + interaction_name + "_interaction_graph.json", 'r') as f:
                data = json.load(f)
                G = nx.Graph()

                for node in data['nodes']:
                    G.add_node(node["id"], atom_type=node["attype"], origin=node["pl"]) 

                for edge in data['edges']:
                    if edge["id1"] != None and edge["id2"] != None:
                        G.add_edge(edge["id1"], edge["id2"], weight= float(edge["length"]))
                        

                for node in data['nodes']:
                    nx.set_node_attributes(G, {node["id"]: node["attype"]}, "atom_type")
                    nx.set_node_attributes(G, {node["id"]: node["pl"]}, "origin")

                
            dataset_dict[interaction_name]["networkx_graph"] = G
            edge_index, edge_weight = create_edge_index(G, weighted=True)

            dataset_dict[interaction_name]["edge_index"] = edge_index
            dataset_dict[interaction_name]["edge_weight"] = edge_weight
            

            num_nodes = G.number_of_nodes()
            
            dataset_dict[interaction_name]["x"] = torch.full((num_nodes, 1), 1.0, dtype=torch.float) #dummy feature

    
    return dataset_dict
        
pli_dataset_dict = generate_pli_dataset_dict_reduced(DATA_PATH + "/dataset/")

data_list = []
for interaction_name in tqdm(pli_dataset_dict):
    edge_weight_sample = None
    edge_weight_sample = pli_dataset_dict[interaction_name]["edge_weight"]
    data_list.append(Data(x = pli_dataset_dict[interaction_name]["x"], edge_index = pli_dataset_dict[interaction_name]["edge_index"], edge_weight = pli_dataset_dict[interaction_name]["edge_weight"], networkx_graph = pli_dataset_dict[interaction_name]["networkx_graph"], interaction_name = interaction_name))

dataset = PLIDataset(".", data_list = data_list)

train_interactions = []
val_interactions = []
core_set_interactions = []
hold_out_interactions = []

with open(DATA_PATH + "pdb_ids/training_set.csv", 'r') as f:
    train_interactions = f.readlines()

train_interactions = [interaction.strip() for interaction in train_interactions]

with open(DATA_PATH + "pdb_ids/validation_set.csv", 'r') as f:
    val_interactions = f.readlines()

val_interactions = [interaction.strip() for interaction in val_interactions]

with open(DATA_PATH + "pdb_ids/core_set.csv", 'r') as f:
    core_set_interactions = f.readlines()

core_set_interactions = [interaction.strip() for interaction in core_set_interactions]

with open(DATA_PATH + "pdb_ids/hold_out_set.csv", 'r') as f:
    hold_out_interactions = f.readlines()

hold_out_interactions = [interaction.strip() for interaction in hold_out_interactions]

train_data = [dataset[i] for i in range(len(dataset)) if dataset[i].interaction_name in train_interactions]
val_data = [dataset[i] for i in range(len(dataset)) if dataset[i].interaction_name in val_interactions]
core_set_data = [dataset[i] for i in range(len(dataset)) if dataset[i].interaction_name in core_set_interactions]
hold_out_data = [dataset[i] for i in range(len(dataset)) if dataset[i].interaction_name in hold_out_interactions]

for affinity_group in AFFINITY_GROUPS:
    
    print("Computing top k for " + affinity_group + " set")

    num_relevant_edge_in_protein_list = {key: [] for key in TOP_K_VALUES}
    num_relevant_edge_in_ligand_list = {key: [] for key in TOP_K_VALUES}
    num_relevant_edge_in_between_list = {key: [] for key in TOP_K_VALUES}
    num_total_relevant_edges_list = {key: [] for key in TOP_K_VALUES}

    num_relevant_absolute_edge_in_protein_list = {key: [] for key in TOP_K_VALUES}
    num_relevant_absolute_edge_in_ligand_list = {key: [] for key in TOP_K_VALUES}
    num_relevant_absolute_edge_in_between_list = {key: [] for key in TOP_K_VALUES}
    num_total_relevant_absolute_list = {key: [] for key in TOP_K_VALUES}

    num_total_edge_in_protein_list = []
    num_total_edge_in_ligand_list = []
    num_total_edge_in_between_list = []
    num_total_edges_in_graph_list = []

    directory = EXPLANATIONS_FOLDER + affinity_group + "/"
    test_interaction_name = None
    test_interaction_index = None

   

    for file in tqdm(os.listdir(directory)):
        test_interaction_name = os.fsdecode(file)
        # if test_interaction_name != SELECTED_INTERACTION_NAME:
        #     continue

        test_interaction_path = directory + test_interaction_name
        if os.path.isdir(test_interaction_path):
            for i, interaction in enumerate(hold_out_data):
                if interaction.interaction_name == test_interaction_name:
                    test_interaction_index = i
                    break
        else:
            continue
        
        

        test_interaction = hold_out_data[test_interaction_index]

        #read phi_edges from file
        phi_edges = []
        

        with open(directory + test_interaction_name + "/" + test_interaction_name + "_statistics.txt", 'r') as f:
            
            shapley_computed = False
            while not shapley_computed:
                line = f.readline()
                if line.strip().startswith("Shapley"):
                    f.readline()
                    lines = f.readlines()
                    for line in lines:
                        phi_edges.append(float(line.strip().split(" ")[-1]))
                    shapley_computed = True

        
        #plotting
        num_bonds = test_interaction.networkx_graph.number_of_edges()


        rdkit_bonds_phi = [0]*num_bonds
        rdkit_bonds = {}

        bonds = dict(test_interaction.networkx_graph.edges())
        bonds = list(bonds.keys())

        for i in range(num_bonds):
            init_atom = bonds[i][0]
            end_atom = bonds[i][1]
            
            rdkit_bonds[(init_atom, end_atom)] = i

        
        
        for i in range(len(phi_edges)):
            phi_value = phi_edges[i]
            init_atom = test_interaction.edge_index[0][i].item()
            end_atom = test_interaction.edge_index[1][i].item()
            
            if (init_atom, end_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(init_atom, end_atom)]
                rdkit_bonds_phi[bond_index] += phi_value
            if (end_atom, init_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(end_atom, init_atom)]
                rdkit_bonds_phi[bond_index] += phi_value
                


        G = test_interaction.networkx_graph
        colors = ["red" if G.nodes[node]["origin"] == "L" else "lightblue" for node in G.nodes]

        num_total_edge_in_protein = 0
        num_total_edge_in_ligand = 0
        num_total_edge_in_between = 0

        atoms_origin = nx.get_node_attributes(G, 'origin')

        for bond in bonds:
            init_atom = bond[0]
            end_atom = bond[1]

            if atoms_origin[init_atom] == "P" and atoms_origin[end_atom] == "P":
                num_total_edge_in_protein += 1
            elif atoms_origin[init_atom] == "L" and atoms_origin[end_atom] == "L":
                num_total_edge_in_ligand += 1
            else:
                num_total_edge_in_between += 1

        num_total_edges_in_graph = num_total_edge_in_protein + num_total_edge_in_ligand + num_total_edge_in_between

        num_total_edge_in_protein_list.append(num_total_edge_in_protein)
        num_total_edge_in_ligand_list.append(num_total_edge_in_ligand)
        num_total_edge_in_between_list.append(num_total_edge_in_between)
        num_total_edges_in_graph_list.append(num_total_edges_in_graph)


        
        with open(directory + test_interaction_name + "/" + test_interaction.interaction_name + "_statistics_top_k_edges.txt", "w+") as f:
            f.write("Top k edges statistics\n\n")

        absolute_phi = np.abs(rdkit_bonds_phi)
        #sort indices according to decreasing phi values
        indices_sorted = np.argsort(-absolute_phi)
        
        for top_k_t in TOP_K_VALUES:

            top_edges = indices_sorted[:top_k_t]

            num_total_top_abs_edges = top_k_t
            num_edge_in_protein = 0
            num_edge_in_ligand = 0
            num_edge_in_between = 0

            atoms_origin = nx.get_node_attributes(G, 'origin')

            edges_to_draw = []
            edges_colors = []
            edges_widths = []
            for bond in bonds:
                init_atom = bond[0]
                end_atom = bond[1]

                bond_index = rdkit_bonds[(init_atom, end_atom)]
                if bond_index in top_edges:
                    if atoms_origin[init_atom] == "P" and atoms_origin[end_atom] == "P":
                        num_edge_in_protein += 1
                        edges_colors.append("darkblue")
                    elif atoms_origin[init_atom] == "L" and atoms_origin[end_atom] == "L":
                        num_edge_in_ligand += 1
                        edges_colors.append("darkred")
                    else:
                        num_edge_in_between += 1
                        edges_colors.append("darkgreen")
                    edges_widths.append(3)
                
                    edges_to_draw.append((init_atom, end_atom))
                else:
                    edges_colors.append("lightgrey") 
                    edges_widths.append(1.5)

            if PLOT:

                #draw graph with important edges
                plt.figure(figsize=(10,10))
                pos = nx.spring_layout(G)

                nx.draw(G, pos=pos, with_labels=True, font_weight='bold', labels=nx.get_node_attributes(G, 'atom_type'), node_color=colors,edge_color=edges_colors, width=edges_widths, edge_cmap=plt.cm.bwr)   

                plt.savefig(directory + test_interaction_name + "/" + test_interaction.interaction_name + "_EdgeSHAPer_top_" + str(top_k_t) + "_edges_full_graph.png", dpi=300)
                
                plt.close()

                #save original graph
                if top_k_t == 25:
                    plt.figure(figsize=(10,10))
                    
                    nx.draw(G, pos=pos, with_labels=True, font_weight='bold', labels=nx.get_node_attributes(G, 'atom_type'), node_color=colors)
                    
                    plt.savefig(directory + test_interaction_name + "/" + test_interaction.interaction_name + "_full_interaction_graph.png", dpi=300)
                    
                    plt.close()
                
            
            with open(directory + test_interaction_name + "/" + test_interaction.interaction_name + "_statistics_top_k_edges.txt", "a") as f:
                f.write("Top " + str(top_k_t) + " relevant edges\n\n")
                if num_total_edge_in_protein == 0:
                    f.write("Number of relevant edges connecting protein pseudo-atoms: 0\n")
                    
                else:
                    f.write("Number of relevant edges connecting protein pseudo-atoms: " + str(num_edge_in_protein) + "\n")
                    f.write("% w.r.t. total number of relevant edges: " + str(round((num_edge_in_protein/num_total_top_abs_edges)*100, 1)) + "%)\n")
                    
                if num_total_edge_in_ligand == 0:
                    f.write("Number of relevant edges connecting ligand pseudo-atoms: 0\n")
                    
                else:
                    f.write("Number of relevant edges connecting ligand pseudo-atoms: " + str(num_edge_in_ligand) + "\n")
                    f.write("% w.r.t. total number of relevant edges: " + str(round((num_edge_in_ligand/num_total_top_abs_edges)*100, 1)) + "%)\n")
                    
                if num_total_edge_in_between == 0:
                    f.write("Number of relevant edges connecting protein and ligand pseudo-atoms: 0\n")
                    
                else:
                    f.write("Number of relevant edges connecting protein and ligand pseudo-atoms: " + str(num_edge_in_between) + "\n")
                    f.write("% w.r.t. total number of relevant edges: " + str(round((num_edge_in_between/num_total_top_abs_edges)*100, 1)) + "%)\n\n")
                    
        
                num_relevant_edge_in_protein_list[top_k_t].append(num_edge_in_protein)
                num_relevant_edge_in_ligand_list[top_k_t].append(num_edge_in_ligand)
                num_relevant_edge_in_between_list[top_k_t].append(num_edge_in_between)
                num_total_relevant_edges_list[top_k_t].append(num_total_top_abs_edges)

    
    with open(directory + "/statistics_top_k_edges.txt", "w+") as f:
        f.write("Top k edges statistics\n\n")

    for top_k_t in TOP_K_VALUES:

        with open(directory + "/statistics_top_k_edges.txt", "a") as f:

            f.write("Top " + str(top_k_t) + " relevant edges\n\n")
           
            f.write("Avg number of relevant edges in protein: " +  str(round(np.mean(num_relevant_edge_in_protein_list[top_k_t]), 3)) + "\n")
            f.write("% w.r.t. total number of relevant edges: " + str(round((np.mean(num_relevant_edge_in_protein_list[top_k_t])/np.mean(num_total_relevant_edges_list[top_k_t]))*100, 1)) + "%\n")
            f.write("Avg number of relevant edges in ligand: " +  str(round(np.mean(num_relevant_edge_in_ligand_list[top_k_t]), 3)) + "\n")
            f.write("% w.r.t. total number of relevant edges: " + str(round((np.mean(num_relevant_edge_in_ligand_list[top_k_t])/np.mean(num_total_relevant_edges_list[top_k_t]))*100, 1)) + "%\n")
            f.write("Avg number of relevant edges in interaction: " + str(round(np.mean(num_relevant_edge_in_between_list[top_k_t]), 3)) + "\n")
            f.write("% w.r.t. total number of relevant edges: " + str(round((np.mean(num_relevant_edge_in_between_list[top_k_t])/np.mean(num_total_relevant_edges_list[top_k_t]))*100, 1)) + "%\n\n")

