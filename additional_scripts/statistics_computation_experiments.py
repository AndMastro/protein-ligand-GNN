### Module implementing top-k edges computation phase ###
### Author: Andrea Mastropietro © All rights reserved ###


import os
from tqdm import tqdm
import json
import yaml

import torch
from torch_geometric.data import Data

import numpy as np
import pandas as pd
import networkx as nx


from src.utils import create_edge_index, PLIDataset



with open("parameters.yml") as paramFile:  
        args = yaml.load(paramFile, Loader=yaml.FullLoader)

DATA_PATH = args["statistics"]["DATA_PATH"]
LOW_BOUND = args["statistics"]["LOW_BOUND"]
HIGH_BOUND = args["statistics"]["HIGH_BOUND"]
MEAN_LOWER_BOUND = args["statistics"]["MEAN_LOWER_BOUND"]
MEAN_UPPER_BOUND = args["statistics"]["MEAN_UPPER_BOUND"]
CLEAN_DATA = args["statistics"]["CLEAN_DATA"]
MIN_AFFINITY = args["statistics"]["MIN_AFFINITY"]
MAX_AFFINITY = args["statistics"]["MAX_AFFINITY"]
EXPLANATIONS_FOLDER = args["statistics"]["EXPLANATIONS_FOLDER"]
TOP_K_VALUES = args["statistics"]["TOP_K_VALUES"]

AFFINITY_GROUPS = ["low affinity", "medium affinity", "high affinity"]

interaction_affinities = None

with open(DATA_PATH + '/interaction_affinities.json', 'r') as fp:
    interaction_affinities = json.load(fp)


affinities_df = pd.DataFrame.from_dict(interaction_affinities, orient='index', columns=['affinity'])

if CLEAN_DATA == True:
    affinities_df = affinities_df[affinities_df['affinity'] >= MIN_AFFINITY]
    affinities_df = affinities_df[affinities_df['affinity'] <= MAX_AFFINITY]


affinities_df = affinities_df.sort_values(by = "affinity", ascending=True)

interaction_affinities = affinities_df.to_dict(orient='index')

def generate_pli_dataset_dict(data_path):

    directory = os.fsencode(data_path)

    dataset_dict = {}
    dirs = os.listdir(directory)
    for file in tqdm(dirs):
        interaction_name = os.fsdecode(file)

        if interaction_name in interaction_affinities:
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


                ## gather label
                dataset_dict[interaction_name]["y"] = torch.FloatTensor([interaction_affinities[interaction_name]["affinity"]])

    
    return dataset_dict
        
pli_dataset_dict = generate_pli_dataset_dict(DATA_PATH + "/dataset/")

data_list = []
for interaction_name in tqdm(pli_dataset_dict):
    edge_weight_sample = None
    edge_weight_sample = pli_dataset_dict[interaction_name]["edge_weight"]
    data_list.append(Data(x = pli_dataset_dict[interaction_name]["x"], y = pli_dataset_dict[interaction_name]["y"], edge_index = pli_dataset_dict[interaction_name]["edge_index"], edge_weight = pli_dataset_dict[interaction_name]["edge_weight"], networkx_graph = pli_dataset_dict[interaction_name]["networkx_graph"], interaction_name = interaction_name))

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

low_affinity_hold_out_data = [hold_out_data[i] for i in range(len(hold_out_data)) if hold_out_data[i].y < LOW_BOUND]
high_affinity_hold_out_data = [hold_out_data[i] for i in range(len(hold_out_data)) if hold_out_data[i].y > HIGH_BOUND]
medium_affinity_hold_out_data = [hold_out_data[i] for i in range(len(hold_out_data)) if (hold_out_data[i].y >= MEAN_LOWER_BOUND and hold_out_data[i].y <= MEAN_UPPER_BOUND)]
# count avg number of protein nodes and ligand nodes in training set
num_protein_nodes = []
num_ligand_nodes = []
total_nodes = []

for interaction in train_data:
    G = interaction.networkx_graph
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_num_protein_nodes = 0
    current_num_ligand_nodes = 0
    num_nodes = G.number_of_nodes()
    for node in G.nodes:
        if atoms_origin[node] == "P":
            current_num_protein_nodes += 1
        elif atoms_origin[node] == "L":
            current_num_ligand_nodes += 1
        else:
            raise Exception("Error: node origin not recognized (P and L are the only valid values)")
    num_protein_nodes.append(current_num_protein_nodes)
    num_ligand_nodes.append(current_num_ligand_nodes)
    total_nodes.append(num_nodes)


avg_num_protein_nodes = sum(num_protein_nodes)/len(num_protein_nodes)
avg_num_ligand_nodes = sum(num_ligand_nodes)/len(num_ligand_nodes)
avg_num_total_nodes = sum(total_nodes)/len(total_nodes)

# print("Avg number of protein nodes in training set: " + str(avg_num_protein_nodes))
# print("Avg number of ligand nodes in training set: " + str(avg_num_ligand_nodes))
print("'%' of protein nodes in training set: " + str(round((avg_num_protein_nodes/avg_num_total_nodes)*100,1)))
print("'%' of ligand nodes in training set: " + str(round((avg_num_ligand_nodes/avg_num_total_nodes)*100,1)))

# count avg number of protein nodes and ligand nodes in validation set
num_protein_nodes = []
num_ligand_nodes = []
total_nodes = []
for interaction in val_data:
    G = interaction.networkx_graph
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_num_protein_nodes = 0
    current_num_ligand_nodes = 0
    num_nodes = G.number_of_nodes()
    for node in G.nodes:
        if atoms_origin[node] == "P":
            current_num_protein_nodes += 1
        elif atoms_origin[node] == "L":
            current_num_ligand_nodes += 1
        else:
            raise Exception("Error: node origin not recognized (P and L are the only valid values)")
    num_protein_nodes.append(current_num_protein_nodes)
    num_ligand_nodes.append(current_num_ligand_nodes)
    total_nodes.append(num_nodes)

avg_num_protein_nodes = sum(num_protein_nodes)/len(num_protein_nodes)
avg_num_ligand_nodes = sum(num_ligand_nodes)/len(num_ligand_nodes)
avg_num_total_nodes = sum(total_nodes)/len(total_nodes)

# print("Avg number of protein nodes in validation set: " + str(avg_num_protein_nodes))
# print("Avg number of ligand nodes in validation set: " + str(avg_num_ligand_nodes))
print("'%' of protein nodes in validation set: " + str(round((avg_num_protein_nodes/avg_num_total_nodes)*100,1)))
print("'%' of ligand nodes in validation set: " + str(round((avg_num_ligand_nodes/avg_num_total_nodes)*100,1)))

# count avg number of protein nodes and ligand nodes in core set
num_protein_nodes = []
num_ligand_nodes = []
total_nodes = []
for interaction in core_set_data:
    G = interaction.networkx_graph
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_num_protein_nodes = 0
    current_num_ligand_nodes = 0
    num_nodes = G.number_of_nodes()
    for node in G.nodes:
        if atoms_origin[node] == "P":
            current_num_protein_nodes += 1
        elif atoms_origin[node] == "L":
            current_num_ligand_nodes += 1
        else:
            raise Exception("Error: node origin not recognized (P and L are the only valid values)")
    num_protein_nodes.append(current_num_protein_nodes)
    num_ligand_nodes.append(current_num_ligand_nodes)
    total_nodes.append(num_nodes)

avg_num_protein_nodes = sum(num_protein_nodes)/len(num_protein_nodes)
avg_num_ligand_nodes = sum(num_ligand_nodes)/len(num_ligand_nodes)
avg_num_total_nodes = sum(total_nodes)/len(total_nodes)

# print("Avg number of protein nodes in core set: " + str(avg_num_protein_nodes))
# print("Avg number of ligand nodes in core set: " + str(avg_num_ligand_nodes))
print("'%' of protein nodes in core set: " + str(round((avg_num_protein_nodes/avg_num_total_nodes)*100,1)))
print("'%' of ligand nodes in core set: " + str(round((avg_num_ligand_nodes/avg_num_total_nodes)*100,1)))

# count avg number of protein nodes and ligand nodes in hold out set
num_protein_nodes = []
num_ligand_nodes = []
total_nodes = []
for interaction in hold_out_data:
    G = interaction.networkx_graph
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_num_protein_nodes = 0
    current_num_ligand_nodes = 0
    num_nodes = G.number_of_nodes()
    for node in G.nodes:
        if atoms_origin[node] == "P":
            current_num_protein_nodes += 1
        elif atoms_origin[node] == "L":
            current_num_ligand_nodes += 1
        else:
            raise Exception("Error: node origin not recognized (P and L are the only valid values)")
    num_protein_nodes.append(current_num_protein_nodes)
    num_ligand_nodes.append(current_num_ligand_nodes)
    total_nodes.append(num_nodes)

avg_num_protein_nodes = sum(num_protein_nodes)/len(num_protein_nodes)
avg_num_ligand_nodes = sum(num_ligand_nodes)/len(num_ligand_nodes)
avg_num_total_nodes = sum(total_nodes)/len(total_nodes)

# print("Avg number of protein nodes in hold out set: " + str(avg_num_protein_nodes))
# print("Avg number of ligand nodes in hold out set: " + str(avg_num_ligand_nodes))
print("'%' of protein nodes in hold-out set: " + str(round((avg_num_protein_nodes/avg_num_total_nodes)*100,1)))
print("'%' of ligand nodes in hold-out set: " + str(round((avg_num_ligand_nodes/avg_num_total_nodes)*100,1)))

# count avg number of protein nodes and ligand nodes in whole dataset
num_protein_nodes = []
num_ligand_nodes = []
total_nodes
for interaction in dataset:
    G = interaction.networkx_graph
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_num_protein_nodes = 0
    current_num_ligand_nodes = 0
    num_nodes = G.number_of_nodes()
    for node in G.nodes:
        if atoms_origin[node] == "P":
            current_num_protein_nodes += 1
        elif atoms_origin[node] == "L":
            current_num_ligand_nodes += 1
        else:
            raise Exception("Error: node origin not recognized (P and L are the only valid values)")
    num_protein_nodes.append(current_num_protein_nodes)
    num_ligand_nodes.append(current_num_ligand_nodes)
    total_nodes.append(num_nodes)

avg_num_protein_nodes = sum(num_protein_nodes)/len(num_protein_nodes)
avg_num_ligand_nodes = sum(num_ligand_nodes)/len(num_ligand_nodes)
avg_num_total_nodes = sum(total_nodes)/len(total_nodes)

# print("Avg number of protein nodes in whole dataset: " + str(avg_num_protein_nodes))
# print("Avg number of ligand nodes in whole dataset: " + str(avg_num_ligand_nodes))
print("'%' of protein nodes in whole dataset set: " + str(round((avg_num_protein_nodes/avg_num_total_nodes)*100,1)))
print("'%' of ligand nodes in whole dataset set: " + str(round((avg_num_ligand_nodes/avg_num_total_nodes)*100,1)))

print("##############################################")
# count avg number of protein nodes and ligand nodes in training set involved in interactions

num_protein_nodes_in_interactions = []
num_ligand_nodes_in_interactions = []
total_nodes_in_interactions = []
for interaction in train_data:
    G = interaction.networkx_graph
    bonds = dict(interaction.networkx_graph.edges())
    bonds = list(bonds.keys())
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_protein_atoms_in_interactions = set()
    current_ligand_atoms_in_interactions = set()

    for bond in bonds:
        init_atom = bond[0]
        end_atom = bond[1]

        if (atoms_origin[init_atom] == "P" and atoms_origin[end_atom] == "L"):
            current_protein_atoms_in_interactions.add(init_atom)
            current_ligand_atoms_in_interactions.add(end_atom)
        elif (atoms_origin[init_atom] == "L" and atoms_origin[end_atom] == "P"):
            current_protein_atoms_in_interactions.add(end_atom)
            current_ligand_atoms_in_interactions.add(init_atom)
        else:
            continue
        
    num_protein_nodes_in_interactions.append(len(current_protein_atoms_in_interactions))
    num_ligand_nodes_in_interactions.append(len(current_ligand_atoms_in_interactions))
    total_nodes_in_interactions.append(len(current_protein_atoms_in_interactions) + len(current_ligand_atoms_in_interactions))

avg_num_protein_nodes_in_interactions = sum(num_protein_nodes_in_interactions)/len(num_protein_nodes_in_interactions)
avg_num_ligand_nodes_in_interactions = sum(num_ligand_nodes_in_interactions)/len(num_ligand_nodes_in_interactions)
avg_num_total_nodes_in_interactions = sum(total_nodes_in_interactions)/len(total_nodes_in_interactions)

# print("Avg number of protein nodes in training set involved in interactions: " + str(avg_num_protein_nodes_in_interactions))
# print("Avg number of ligand nodes in training set involved in interactions: " + str(avg_num_ligand_nodes_in_interactions))
print("'%' of protein nodes in training set involved in interactions: " + str(round((avg_num_protein_nodes_in_interactions/avg_num_total_nodes_in_interactions)*100,1)))
print("'%' of ligand nodes in training set involved in interactions: " + str(round((avg_num_ligand_nodes_in_interactions/avg_num_total_nodes_in_interactions)*100,1)))
# print("'%' of ligand nodes in training set involved in interactions: " + str(avg_num_ligand_nodes_in_interactions/avg_num_total_nodes_in_interactions))

# count avg number of protein nodes and ligand nodes in validation set involved in interactions

num_protein_nodes_in_interactions = []
num_ligand_nodes_in_interactions = []
total_nodes_in_interactions = []
for interaction in val_data:
    G = interaction.networkx_graph
    bonds = dict(interaction.networkx_graph.edges())
    bonds = list(bonds.keys())
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_protein_atoms_in_interactions = set()
    current_ligand_atoms_in_interactions = set()
    for bond in bonds:
        init_atom = bond[0]
        end_atom = bond[1]

        if (atoms_origin[init_atom] == "P" and atoms_origin[end_atom] == "L"):
            current_protein_atoms_in_interactions.add(init_atom)
            current_ligand_atoms_in_interactions.add(end_atom)
        elif (atoms_origin[init_atom] == "L" and atoms_origin[end_atom] == "P"):
            current_protein_atoms_in_interactions.add(end_atom)
            current_ligand_atoms_in_interactions.add(init_atom)
        else:
            continue
        
    num_protein_nodes_in_interactions.append(len(current_protein_atoms_in_interactions))
    num_ligand_nodes_in_interactions.append(len(current_ligand_atoms_in_interactions))
    total_nodes_in_interactions.append(len(current_protein_atoms_in_interactions) + len(current_ligand_atoms_in_interactions))

avg_num_protein_nodes_in_interactions = sum(num_protein_nodes_in_interactions)/len(num_protein_nodes_in_interactions)
avg_num_ligand_nodes_in_interactions = sum(num_ligand_nodes_in_interactions)/len(num_ligand_nodes_in_interactions)
avg_num_total_nodes_in_interactions = sum(total_nodes_in_interactions)/len(total_nodes_in_interactions)

# print("Avg number of protein nodes in validation set involved in interactions: " + str(avg_num_protein_nodes_in_interactions))
# print("Avg number of ligand nodes in validation set involved in interactions: " + str(avg_num_ligand_nodes_in_interactions))
print("'%' of protein nodes in validation set involved in interactions: " + str(round((avg_num_protein_nodes_in_interactions/avg_num_total_nodes_in_interactions)*100,1)))
print("'%' of ligand nodes in validation set involved in interactions: " + str(round((avg_num_ligand_nodes_in_interactions/avg_num_total_nodes_in_interactions)*100,1)))

# count avg number of protein nodes and ligand nodes in core set involved in interactions

num_protein_nodes_in_interactions = []
num_ligand_nodes_in_interactions = []
total_nodes_in_interactions = []
for interaction in core_set_data:
    G = interaction.networkx_graph
    bonds = dict(interaction.networkx_graph.edges())
    bonds = list(bonds.keys())
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_protein_atoms_in_interactions = set()
    current_ligand_atoms_in_interactions = set()
    for bond in bonds:
        init_atom = bond[0]
        end_atom = bond[1]

        if (atoms_origin[init_atom] == "P" and atoms_origin[end_atom] == "L"):
            current_protein_atoms_in_interactions.add(init_atom)
            current_ligand_atoms_in_interactions.add(end_atom)
        elif (atoms_origin[init_atom] == "L" and atoms_origin[end_atom] == "P"):
            current_protein_atoms_in_interactions.add(end_atom)
            current_ligand_atoms_in_interactions.add(init_atom)
        else:
            continue
        
    num_protein_nodes_in_interactions.append(len(current_protein_atoms_in_interactions))
    num_ligand_nodes_in_interactions.append(len(current_ligand_atoms_in_interactions))
    total_nodes_in_interactions.append(len(current_protein_atoms_in_interactions) + len(current_ligand_atoms_in_interactions))

avg_num_protein_nodes_in_interactions = sum(num_protein_nodes_in_interactions)/len(num_protein_nodes_in_interactions)
avg_num_ligand_nodes_in_interactions = sum(num_ligand_nodes_in_interactions)/len(num_ligand_nodes_in_interactions)
avg_num_total_nodes_in_interactions = sum(total_nodes_in_interactions)/len(total_nodes_in_interactions)

# print("Avg number of protein nodes in core set involved in interactions: " + str(avg_num_protein_nodes_in_interactions))
# print("Avg number of ligand nodes in core set involved in interactions: " + str(avg_num_ligand_nodes_in_interactions))
print("'%' of protein nodes in core set involved in interactions: " + str(round((avg_num_protein_nodes_in_interactions/avg_num_total_nodes_in_interactions)*100,1)))
print("'%' of ligand nodes in core set involved in interactions: " + str(round((avg_num_ligand_nodes_in_interactions/avg_num_total_nodes_in_interactions)*100,1)))

# count avg number of protein nodes and ligand nodes in hold out set involved in interactions

num_protein_nodes_in_interactions = []
num_ligand_nodes_in_interactions = []
total_nodes_in_interactions = []
for interaction in hold_out_data:
    G = interaction.networkx_graph
    bonds = dict(interaction.networkx_graph.edges())
    bonds = list(bonds.keys())
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_protein_atoms_in_interactions = set()
    current_ligand_atoms_in_interactions = set()
    for bond in bonds:
        init_atom = bond[0]
        end_atom = bond[1]

        if (atoms_origin[init_atom] == "P" and atoms_origin[end_atom] == "L"):
            current_protein_atoms_in_interactions.add(init_atom)
            current_ligand_atoms_in_interactions.add(end_atom)
        elif (atoms_origin[init_atom] == "L" and atoms_origin[end_atom] == "P"):
            current_protein_atoms_in_interactions.add(end_atom)
            current_ligand_atoms_in_interactions.add(init_atom)
        else:
            continue
        
    num_protein_nodes_in_interactions.append(len(current_protein_atoms_in_interactions))
    num_ligand_nodes_in_interactions.append(len(current_ligand_atoms_in_interactions))
    total_nodes_in_interactions.append(len(current_protein_atoms_in_interactions) + len(current_ligand_atoms_in_interactions))

avg_num_protein_nodes_in_interactions = sum(num_protein_nodes_in_interactions)/len(num_protein_nodes_in_interactions)
avg_num_ligand_nodes_in_interactions = sum(num_ligand_nodes_in_interactions)/len(num_ligand_nodes_in_interactions)
avg_num_total_nodes_in_interactions = sum(total_nodes_in_interactions)/len(total_nodes_in_interactions)

# print("Avg number of protein nodes in hold out set involved in interactions: " + str(avg_num_protein_nodes_in_interactions))
# print("Avg number of ligand nodes in hold out set involved in interactions: " + str(avg_num_ligand_nodes_in_interactions))
print("'%' of protein nodes in hold-out set involved in interactions: " + str(round((avg_num_protein_nodes_in_interactions/avg_num_total_nodes_in_interactions)*100,1)))
print("'%' of ligand nodes in hold-out set involved in interactions: " + str(round((avg_num_ligand_nodes_in_interactions/avg_num_total_nodes_in_interactions)*100,1)))

# count avg number of protein nodes and ligand nodes in whole dataset involved in interactions

num_protein_nodes_in_interactions = []
num_ligand_nodes_in_interactions = []
total_nodes_in_interactions = []
for interaction in dataset:
    G = interaction.networkx_graph
    bonds = dict(interaction.networkx_graph.edges())
    bonds = list(bonds.keys())
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_protein_atoms_in_interactions = set()
    current_ligand_atoms_in_interactions = set()
    for bond in bonds:
        init_atom = bond[0]
        end_atom = bond[1]

        if (atoms_origin[init_atom] == "P" and atoms_origin[end_atom] == "L"):
            current_protein_atoms_in_interactions.add(init_atom)
            current_ligand_atoms_in_interactions.add(end_atom)
        elif (atoms_origin[init_atom] == "L" and atoms_origin[end_atom] == "P"):
            current_protein_atoms_in_interactions.add(end_atom)
            current_ligand_atoms_in_interactions.add(init_atom)
        else:
            continue
        
    num_protein_nodes_in_interactions.append(len(current_protein_atoms_in_interactions))
    num_ligand_nodes_in_interactions.append(len(current_ligand_atoms_in_interactions))
    total_nodes_in_interactions.append(len(current_protein_atoms_in_interactions) + len(current_ligand_atoms_in_interactions))

avg_num_protein_nodes_in_interactions = sum(num_protein_nodes_in_interactions)/len(num_protein_nodes_in_interactions)
avg_num_ligand_nodes_in_interactions = sum(num_ligand_nodes_in_interactions)/len(num_ligand_nodes_in_interactions)
avg_num_total_nodes_in_interactions = sum(total_nodes_in_interactions)/len(total_nodes_in_interactions)


# print("Avg number of protein nodes in whole dataset involved in interactions: " + str(avg_num_protein_nodes_in_interactions))
# print("Avg number of ligand nodes in whole dataset involved in interactions: " + str(avg_num_ligand_nodes_in_interactions))
print("'%' of protein nodes in whole dataset set involved in interactions: " + str(round((avg_num_protein_nodes_in_interactions/avg_num_total_nodes_in_interactions)*100,1)))
print("'%' of ligand nodes in whole dataset set involved in interactions: " + str(round((avg_num_ligand_nodes_in_interactions/avg_num_total_nodes_in_interactions)*100,1)))

print("##############################################")

# avg number of protein, ligand and interaction edges in hold out set
relative_fraction_of_edges = {"protein": 0, "ligand": 0, "interaction": 0}

num_protein_edges = []
num_ligand_edges = []
num_interaction_edges = []
num_intramolecular_edges = []
num_total_edge_in_graph = []
for interaction in hold_out_data:
    G = interaction.networkx_graph
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_num_protein_edges = 0
    current_num_ligand_edges = 0
    current_num_interaction_edges = 0
   
    for edge in G.edges:
        init_atom = edge[0]
        end_atom = edge[1]

        if atoms_origin[init_atom] == "P" and atoms_origin[end_atom] == "P":
            current_num_protein_edges += 1
        elif atoms_origin[init_atom] == "L" and atoms_origin[end_atom] == "L":
            current_num_ligand_edges += 1
        else:
            current_num_interaction_edges += 1
    
    num_protein_edges.append(current_num_protein_edges)
    num_ligand_edges.append(current_num_ligand_edges)
    num_interaction_edges.append(current_num_interaction_edges)

    num_intramolecular_edges.append(current_num_protein_edges + current_num_ligand_edges)
    
    num_total_edge_in_graph.append(current_num_protein_edges + current_num_ligand_edges + current_num_interaction_edges)

avg_num_protein_edges = sum(num_protein_edges)/len(num_protein_edges)
avg_num_ligand_edges = sum(num_ligand_edges)/len(num_ligand_edges)
avg_num_interaction_edges = sum(num_interaction_edges)/len(num_interaction_edges)

avg_num_intramolecular_edges = sum(num_intramolecular_edges)/len(num_intramolecular_edges)
avg_num_total_edges_in_graph = sum(num_total_edge_in_graph)/len(num_total_edge_in_graph)

# print("Avg number of protein edges in hold out set: " + str(avg_num_protein_edges))
# print("Avg number of ligand edges in hold out set: " + str(avg_num_ligand_edges))
# print("Avg number of interaction edges in hold out set: " + str(avg_num_interaction_edges))
print("Relative '%' of protein edges: " + str(round((avg_num_protein_edges/avg_num_total_edges_in_graph)*100,1)))
relative_fraction_of_edges["protein"] = avg_num_protein_edges/avg_num_total_edges_in_graph
print("Relative '%' of ligand edges: " + str(round((avg_num_ligand_edges/avg_num_total_edges_in_graph)*100,1)))
relative_fraction_of_edges["ligand"] = avg_num_ligand_edges/avg_num_total_edges_in_graph
print("Relative '%' of intermolecular (interaction) edges: " + str(round((avg_num_interaction_edges/avg_num_total_edges_in_graph)*100,1)))
relative_fraction_of_edges["interaction"] = avg_num_interaction_edges/avg_num_total_edges_in_graph
print("Relative '%' of intramolecular (protein or ligand) edges: " + str(round((avg_num_intramolecular_edges/avg_num_total_edges_in_graph)*100,1)))
print("##############################################")

# percentage_total_edges = {"low affinity": [], "medium affinity": [], "high affinity": []}

# avg number of protein, ligand and interaction edges in for each affinity level in hold out set

num_protein_edges = []
num_ligand_edges = []
num_interaction_edges = []
num_intramolecular_edges = []
num_total_edge_in_graph = []
for interaction in low_affinity_hold_out_data:
    G = interaction.networkx_graph
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_num_protein_edges = 0
    current_num_ligand_edges = 0
    current_num_interaction_edges = 0
   
    for edge in G.edges:
        init_atom = edge[0]
        end_atom = edge[1]

        if atoms_origin[init_atom] == "P" and atoms_origin[end_atom] == "P":
            current_num_protein_edges += 1
        elif atoms_origin[init_atom] == "L" and atoms_origin[end_atom] == "L":
            current_num_ligand_edges += 1
        else:
            current_num_interaction_edges += 1
    
    num_protein_edges.append(current_num_protein_edges)
    num_ligand_edges.append(current_num_ligand_edges)
    num_interaction_edges.append(current_num_interaction_edges)

    num_intramolecular_edges.append(current_num_protein_edges + current_num_ligand_edges)
    
    num_total_edge_in_graph.append(current_num_protein_edges + current_num_ligand_edges + current_num_interaction_edges)

avg_num_protein_edges = sum(num_protein_edges)/len(num_protein_edges)
avg_num_ligand_edges = sum(num_ligand_edges)/len(num_ligand_edges)
avg_num_interaction_edges = sum(num_interaction_edges)/len(num_interaction_edges)

avg_num_intramolecular_edges = sum(num_intramolecular_edges)/len(num_intramolecular_edges)
avg_num_total_edges_in_graph = sum(num_total_edge_in_graph)/len(num_total_edge_in_graph)

print("Low affinity hold out set")
# print("Avg number of protein edges in low affinity hold out set: " + str(avg_num_protein_edges))
# print("Avg number of ligand edges in low affinity hold out set: " + str(avg_num_ligand_edges))
# print("Avg number of interaction edges in low affinity hold out set: " + str(avg_num_interaction_edges))
print("Relative '%' of protein edges: " + str(round((avg_num_protein_edges/avg_num_total_edges_in_graph)*100,1)))
print("Relative '%' of ligand edges: " + str(round((avg_num_ligand_edges/avg_num_total_edges_in_graph)*100,1)))
print("Relative '%' of interaction (intermolecular) edges: " + str(round((avg_num_interaction_edges/avg_num_total_edges_in_graph)*100,1)))
print("Relative '%' of intramolecular (protein or ligand) edges: " + str(round((avg_num_intramolecular_edges/avg_num_total_edges_in_graph)*100,1)))
# print("Relative '%' of ligand edges: " + str(avg_num_ligand_edges/avg_num_total_edges_in_graph))
# print("Relative '%' of intermolecular (interaction) edges: " + str(avg_num_interaction_edges/avg_num_total_edges_in_graph))
# print("Relative '%' of intramolecular (protein or ligand) edges: " + str(avg_num_intramolecular_edges/avg_num_total_edges_in_graph))

print("##############################################")

num_protein_edges = []
num_ligand_edges = []
num_interaction_edges = []
num_intramolecular_edges = []
num_total_edge_in_graph = []
for interaction in medium_affinity_hold_out_data:
    G = interaction.networkx_graph
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_num_protein_edges = 0
    current_num_ligand_edges = 0
    current_num_interaction_edges = 0
   
    for edge in G.edges:
        init_atom = edge[0]
        end_atom = edge[1]

        if atoms_origin[init_atom] == "P" and atoms_origin[end_atom] == "P":
            current_num_protein_edges += 1
        elif atoms_origin[init_atom] == "L" and atoms_origin[end_atom] == "L":
            current_num_ligand_edges += 1
        else:
            current_num_interaction_edges += 1
    
    num_protein_edges.append(current_num_protein_edges)
    num_ligand_edges.append(current_num_ligand_edges)
    num_interaction_edges.append(current_num_interaction_edges)

    num_intramolecular_edges.append(current_num_protein_edges + current_num_ligand_edges)
    
    num_total_edge_in_graph.append(current_num_protein_edges + current_num_ligand_edges + current_num_interaction_edges)

avg_num_protein_edges = sum(num_protein_edges)/len(num_protein_edges)
avg_num_ligand_edges = sum(num_ligand_edges)/len(num_ligand_edges)
avg_num_interaction_edges = sum(num_interaction_edges)/len(num_interaction_edges)

avg_num_intramolecular_edges = sum(num_intramolecular_edges)/len(num_intramolecular_edges)
avg_num_total_edges_in_graph = sum(num_total_edge_in_graph)/len(num_total_edge_in_graph)

print("Medium affinity hold out set")
# print("Avg number of protein edges in medium affinity hold out set: " + str(avg_num_protein_edges))
# print("Avg number of ligand edges in medium affinity hold out set: " + str(avg_num_ligand_edges))
# print("Avg number of interaction edges in medium affinity hold out set: " + str(avg_num_interaction_edges))
print("Relative '%' of protein edges: " + str(round((avg_num_protein_edges/avg_num_total_edges_in_graph)*100,1)))
print("Relative '%' of ligand edges: " + str(round((avg_num_ligand_edges/avg_num_total_edges_in_graph)*100,1)))
print("Relative '%' of interaction (intermolecular) edges: " + str(round((avg_num_interaction_edges/avg_num_total_edges_in_graph)*100,1)))
print("Relative '%' of intramolecular (protein or ligand) edges: " + str(round((avg_num_intramolecular_edges/avg_num_total_edges_in_graph)*100,1)))
# print("Relative '%' of ligand edges: " + str(avg_num_ligand_edges/avg_num_total_edges_in_graph))
# print("Relative '%' of intermolecular (interaction) edges: " + str(avg_num_interaction_edges/avg_num_total_edges_in_graph))
# print("Relative '%' of intramolecular (protein or ligand) edges: " + str(avg_num_intramolecular_edges/avg_num_total_edges_in_graph))

print("##############################################")

num_protein_edges = []
num_ligand_edges = []
num_interaction_edges = []
num_intramolecular_edges = []
num_total_edge_in_graph = []
for interaction in high_affinity_hold_out_data:
    G = interaction.networkx_graph
    atoms_origin = nx.get_node_attributes(G, 'origin')
    current_num_protein_edges = 0
    current_num_ligand_edges = 0
    current_num_interaction_edges = 0
   
    for edge in G.edges:
        init_atom = edge[0]
        end_atom = edge[1]

        if atoms_origin[init_atom] == "P" and atoms_origin[end_atom] == "P":
            current_num_protein_edges += 1
        elif atoms_origin[init_atom] == "L" and atoms_origin[end_atom] == "L":
            current_num_ligand_edges += 1
        else:
            current_num_interaction_edges += 1
    
    num_protein_edges.append(current_num_protein_edges)
    num_ligand_edges.append(current_num_ligand_edges)
    num_interaction_edges.append(current_num_interaction_edges)

    num_intramolecular_edges.append(current_num_protein_edges + current_num_ligand_edges)
    
    num_total_edge_in_graph.append(current_num_protein_edges + current_num_ligand_edges + current_num_interaction_edges)

avg_num_protein_edges = sum(num_protein_edges)/len(num_protein_edges)
avg_num_ligand_edges = sum(num_ligand_edges)/len(num_ligand_edges)
avg_num_interaction_edges = sum(num_interaction_edges)/len(num_interaction_edges)

avg_num_intramolecular_edges = sum(num_intramolecular_edges)/len(num_intramolecular_edges)
avg_num_total_edges_in_graph = sum(num_total_edge_in_graph)/len(num_total_edge_in_graph)

# print("Avg number of protein edges in high affinity hold out set: " + str(avg_num_protein_edges))
# print("Avg number of ligand edges in high affinity hold out set: " + str(avg_num_ligand_edges))
# print("Avg number of interaction edges in high affinity hold out set: " + str(avg_num_interaction_edges))
print("High affinity hold out set")
print("Relative '%' of protein edges: " + str(round((avg_num_protein_edges/avg_num_total_edges_in_graph)*100, 1))) 
print("Relative '%' of ligand edges: " + str(round((avg_num_ligand_edges/avg_num_total_edges_in_graph)*100, 1)))
print("Relative '%' of interaction (intermolecular) edges: " + str(round((avg_num_interaction_edges/avg_num_total_edges_in_graph)*100, 1)))
print("Relative '%' of intramolecular (protein or ligand) edges: " + str(round((avg_num_intramolecular_edges/avg_num_total_edges_in_graph)*100,1)))

# import sys        
# sys.exit(0)    

print(TOP_K_VALUES)
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



        absolute_phi = np.abs(rdkit_bonds_phi)
        #sort indices according to decreasing phi values
        indices_sorted = np.argsort(-absolute_phi)
        
        for top_k_t in TOP_K_VALUES:
            
            top_edges = indices_sorted[:top_k_t]

            # print("top edges: " + str(top_edges))
            
            # print("len top edges: " + str(len(top_edges)))

            # print("bonds: " + str(rdkit_bonds))
            # sys.exit()
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
                    # print("CATRO'")
                    edges_colors.append("lightgrey") 
                    edges_widths.append(1.5)

                
            # print("num_edge_in_protein: " + str(num_edge_in_protein))
            # print("num_edge_in_ligand: " + str(num_edge_in_ligand))
            # print("num_edge_in_between: " + str(num_edge_in_between))
            # print("num_total_top_abs_edges: " + str(num_total_top_abs_edges))
            # print("num_total_edges_in_graph: " + str(num_edge_in_protein + num_edge_in_ligand + num_edge_in_between))
            # sys.exit()
            num_relevant_edge_in_protein_list[top_k_t].append(num_edge_in_protein)
            num_relevant_edge_in_ligand_list[top_k_t].append(num_edge_in_ligand)
            num_relevant_edge_in_between_list[top_k_t].append(num_edge_in_between)
            num_total_relevant_edges_list[top_k_t].append(num_total_top_abs_edges)

    
   

    for top_k_t in TOP_K_VALUES:

        

        print("Top " + str(top_k_t) + " relevant edges\n\n")
        
        print("Avg number of relevant edges in protein: " +  str(round(np.mean(num_relevant_edge_in_protein_list[top_k_t]), 3)) + "\n")
        print("% w.r.t. total number of relevant edges: " + str(round((np.mean(num_relevant_edge_in_protein_list[top_k_t])/np.mean(num_total_relevant_edges_list[top_k_t]))*100, 1)) + "%\n")
        print("Avg number of relevant edges in ligand: " +  str(round(np.mean(num_relevant_edge_in_ligand_list[top_k_t]), 3)) + "\n")
        print("% w.r.t. total number of relevant edges: " + str(round((np.mean(num_relevant_edge_in_ligand_list[top_k_t])/np.mean(num_total_relevant_edges_list[top_k_t]))*100, 1)) + "%\n")
        print("Avg number of relevant edges in interaction: " + str(round(np.mean(num_relevant_edge_in_between_list[top_k_t]), 3)) + "\n")
        print("% w.r.t. total number of relevant edges: " + str(round((np.mean(num_relevant_edge_in_between_list[top_k_t])/np.mean(num_total_relevant_edges_list[top_k_t]))*100, 1)) + "%\n\n")

        print("Relative fraction of top " + str(top_k_t) + " edges for ((topk/25) /(category/total))  " + "\n\n")
        
        print("Protein edges: " + str(round((np.mean(num_relevant_edge_in_protein_list[top_k_t])/np.mean(num_total_relevant_edges_list[top_k_t]))/relative_fraction_of_edges["protein"], 3)) + "\n")
        print("Ligand edges: " + str(round((np.mean(num_relevant_edge_in_ligand_list[top_k_t])/np.mean(num_total_relevant_edges_list[top_k_t]))/relative_fraction_of_edges["ligand"], 3)) + "\n")
        print("Interaction edges: " + str(round((np.mean(num_relevant_edge_in_between_list[top_k_t])/np.mean(num_total_relevant_edges_list[top_k_t]))/relative_fraction_of_edges["interaction"], 3)) + "\n\n")

# print("Relative fraction of top " + str(top_k_t) + " edges for ((topk/25) /(category/total))  " + "\n\n")
        
# #print("Avg number of relevant edges in protein: " +  str(round(np.mean(num_relevant_edge_in_protein_list[top_k_t]), 3)) + "\n")
# print("Protein edges: " + str(round(((np.mean(num_relevant_edge_in_protein_list[top_k_t])/np.mean(num_total_relevant_edges_list[top_k_t]))/relative_fraction_of_edges["protein"])*100, 1)) + "%\n")
# print("Protein edges prova: " + str(round((np.mean(num_relevant_edge_in_protein_list[top_k_t])/np.mean(num_total_relevant_edges_list[top_k_t]))*100, 1)) + "%\n")
# print(num_relevant_edge_in_protein_list)
# #print("Avg number of relevant edges in ligand: " +  str(round(np.mean(num_relevant_edge_in_ligand_list[top_k_t]), 3)) + "\n")
# print("Ligand edges: " + str(round(((np.mean(num_relevant_edge_in_ligand_list[top_k_t])/np.mean(num_total_relevant_edges_list[top_k_t]))/relative_fraction_of_edges["ligand"])*100, 1)) + "%\n")
# #print("Avg number of relevant edges in interaction: " + str(round(np.mean(num_relevant_edge_in_between_list[top_k_t]), 3)) + "\n")
# print("Interaction edges: " + str(round(((np.mean(num_relevant_edge_in_between_list[top_k_t])/np.mean(num_total_relevant_edges_list[top_k_t]))/relative_fraction_of_edges["interaction"])*100, 1)) + "%\n\n")