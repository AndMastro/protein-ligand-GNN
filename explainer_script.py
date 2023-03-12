### Module implementing explanation phase ###
### Author: Andrea Mastropietro © All rights reserved ###

import os
from tqdm import tqdm
import json
import yaml

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler


import json
import networkx as nx
import pandas as pd

from src.utils import create_edge_index, PLIDataset, set_all_seeds, GCN, GraphSAGE, GAT, GIN, GINE, GC_GNN
from src.edgeshaper import Edgeshaper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Working on device: ", device)

with open("parameters.yml") as paramFile:  
        args = yaml.load(paramFile, Loader=yaml.FullLoader)

DATA_PATH = args["explainer"]["DATA_PATH"]
SAVE_FOLDER = args["explainer"]["SAVE_FOLDER"]

CLEAN_DATA = args["explainer"]["CLEAN_DATA"]
MIN_AFFINITY = args["explainer"]["MIN_AFFINITY"]
MAX_AFFINITY = args["explainer"]["MAX_AFFINITY"]
NUM_CLASSES = 1 #set it up to 1 since we are facing a regression problem

MEAN_LOWER_BOUND = args["explainer"]["MEAN_LOWER_BOUND"]
MEAN_UPPER_BOUND = args["explainer"]["MEAN_UPPER_BOUND"]
LOW_BOUND = args["explainer"]["LOW_BOUND"]
HIGH_BOUND = args["explainer"]["HIGH_BOUND"]

MODEL_NAME = args["explainer"]["GNN_MODEL"]
GNN = GCN if MODEL_NAME == "GCN" else GraphSAGE if MODEL_NAME == "GraphSAGE" else GAT if MODEL_NAME == "GAT" else GIN if MODEL_NAME == "GIN" else GINE if MODEL_NAME == "GINE" else GC_GNN if MODEL_NAME == "GC_GNN" else None
MODEL_PATH = args["explainer"]["MODEL_PATH"]

print("Using model: ", MODEL_NAME)

EDGE_WEIGHT = args["explainer"]["EDGE_WEIGHT"]
SCALING = args["explainer"]["SCALING"]
BATCH_SIZE = args["explainer"]["BATCH_SIZE"]
LEARNING_RATE = float(args["explainer"]["LEARNING_RATE"])
WEIGHT_DECAY = float(args["explainer"]["WEIGHT_DECAY"])

SEED = args["explainer"]["SEED"]
HIDDEN_CHANNELS = args["explainer"]["HIDDEN_CHANNELS"]
EPOCHS = args["explainer"]["EPOCHS"]
NODE_FEATURES = args["explainer"]["NODE_FEATURES"] #if False, use dummy features (1)
AFFINITY_SET = args["explainer"]["AFFINITY_SET"] 

assert(AFFINITY_SET == "low" or AFFINITY_SET == "high" or AFFINITY_SET == "medium")

print("Explaining affinity set: ", AFFINITY_SET)

SAMPLES_TO_EXPLAIN = args["explainer"]["SAMPLES_TO_EXPLAIN"]

set_all_seeds(SEED)


interaction_affinities = None

with open(DATA_PATH + '/interaction_affinities.json', 'r') as fp:
    interaction_affinities = json.load(fp)



affinities_df = pd.DataFrame.from_dict(interaction_affinities, orient='index', columns=['affinity'])

if CLEAN_DATA == True:
    affinities_df = affinities_df[affinities_df['affinity'] >= MIN_AFFINITY]
    affinities_df = affinities_df[affinities_df['affinity'] <= MAX_AFFINITY]

vals_cleaned = list(affinities_df['affinity'])
mean_interaction_affinity_no_outliers = np.mean(vals_cleaned)

affinities_df = affinities_df.sort_values(by = "affinity", ascending=True)

interaction_affinities = affinities_df.to_dict(orient='index')

descriptors_interaction_dict = None
num_node_features = 0
if NODE_FEATURES:
    descriptors_interaction_dict = {}
    descriptors_interaction_dict["CA"] = [1, 0, 0, 0, 0, 0, 0, 0]
    descriptors_interaction_dict["NZ"] = [0, 1, 0, 0, 0, 0, 0, 0]
    descriptors_interaction_dict["N"] = [0, 0, 1, 0, 0, 0, 0, 0]
    descriptors_interaction_dict["OG"] = [0, 0, 0, 1, 0, 0, 0, 0]
    descriptors_interaction_dict["O"] = [0, 0, 0, 0, 1, 0, 0, 0]
    descriptors_interaction_dict["CZ"] = [0, 0, 0, 0, 0, 1, 0, 0]
    descriptors_interaction_dict["OD1"] = [0, 0, 0, 0, 0, 0, 1, 0]
    descriptors_interaction_dict["ZN"] = [0, 0, 0, 0, 0, 0, 0, 1]
    num_node_features = len(descriptors_interaction_dict["CA"])

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
                   
                    if not NODE_FEATURES:
                        dataset_dict[interaction_name]["x"] = torch.full((num_nodes, 1), 1.0, dtype=torch.float) #dummy feature
                    else:
                        dataset_dict[interaction_name]["x"] = torch.zeros((num_nodes, num_node_features), dtype=torch.float)
                        for node in G.nodes:
                            
                            dataset_dict[interaction_name]["x"][node] = torch.tensor(descriptors_interaction_dict[G.nodes[node]["atom_type"]], dtype=torch.float)
                        

                    ## gather label
                    dataset_dict[interaction_name]["y"] = torch.FloatTensor([interaction_affinities[interaction_name]["affinity"]])

        
        return dataset_dict
        
pli_dataset_dict = generate_pli_dataset_dict(DATA_PATH + "/dataset/")


if SCALING:
    first_level = [pli_dataset_dict[key]["edge_weight"] for key in pli_dataset_dict]
    second_level = [item for sublist in first_level for item in sublist]
    if MODEL_NAME == "GCN":
        transformer = MinMaxScaler().fit(np.array(second_level).reshape(-1, 1))
    else:
        transformer = RobustScaler().fit(np.array(second_level).reshape(-1, 1))
    for key in tqdm(pli_dataset_dict):
        scaled_weights = transformer.transform(np.array(pli_dataset_dict[key]["edge_weight"]).reshape(-1, 1))
        scaled_weights = [x[0] for x in scaled_weights]
        pli_dataset_dict[key]["edge_weight"] = torch.FloatTensor(scaled_weights)
        
data_list = []
for interaction_name in tqdm(pli_dataset_dict):
    edge_weight_sample = None
    if EDGE_WEIGHT:
        edge_weight_sample = pli_dataset_dict[interaction_name]["edge_weight"]
    data_list.append(Data(x = pli_dataset_dict[interaction_name]["x"], edge_index = pli_dataset_dict[interaction_name]["edge_index"], edge_weight = edge_weight_sample, y = pli_dataset_dict[interaction_name]["y"], networkx_graph = pli_dataset_dict[interaction_name]["networkx_graph"], interaction_name = interaction_name))


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

#alternative here is to comment the shuffle out and use shuffling in the dataloader
rng = np.random.default_rng(seed = SEED)
rng.shuffle(train_data)
rng.shuffle(val_data)
rng.shuffle(core_set_data)
rng.shuffle(hold_out_data)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
core_set_loader = DataLoader(core_set_data, batch_size=BATCH_SIZE)
hold_out_loader = DataLoader(hold_out_data, batch_size=BATCH_SIZE)


model = GNN(node_features_dim = dataset[0].x.shape[1], num_classes = NUM_CLASSES, hidden_channels=HIDDEN_CHANNELS).to(device)

lr = LEARNING_RATE

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.MSELoss()
epochs = EPOCHS

def train():
        model.train()

        for data in train_loader:  
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch, edge_weight = data.edge_weight)  
            
            loss = torch.sqrt(criterion(torch.squeeze(out), data.y))  
        
            loss.backward()  
            optimizer.step()  
            optimizer.zero_grad()  

def test(loader):
    model.eval()

    sum_loss = 0
    for data in loader: 
        data = data.to(device)
        
        out = model(data.x, data.edge_index, data.batch, edge_weight = data.edge_weight)  
        
        if  data.y.shape[0] == 1:
            loss = torch.sqrt(criterion(torch.squeeze(out, 1), data.y))
        else:
            loss = torch.sqrt(criterion(torch.squeeze(out), data.y)) * data.y.shape[0]
        sum_loss += loss.item()
        
    return sum_loss / len(loader.dataset) 


#load model
model.load_state_dict(torch.load(MODEL_PATH)) 
model.to(device)
model

core_set_rmse = test(core_set_loader)    
print(f'Core set 2016 RMSE with loaded model: {core_set_rmse:.4f}')

hold_out_set_rmse = test(hold_out_loader)    
print(f'Hold out set 2019 RMSE with loaded model: {hold_out_set_rmse:.4f}')

#explanation phase

num_all_test_interactions = len(hold_out_data)
rng = np.random.default_rng(seed=SEED)
all_test_interaction_indices = np.array(range(num_all_test_interactions))
rng.shuffle(all_test_interaction_indices)

num_edge_in_protein_list = []
num_edge_in_ligand_list = []
num_edge_in_between_list = []
num_total_relevant_edges_list = []

num_edge_in_protein_abs_list = []
num_edge_in_ligand_abs_list = []
num_edge_in_between_abs_list = []
num_total_relevant_edges_abs_list = []

num_total_edge_in_protein_list = []
num_total_edge_in_ligand_list = []
num_total_edge_in_between_list = []
num_total_edges_in_graph_list = []

num_pert_pos_edges_in_protein_list = []
num_pert_pos_edges_in_ligand_list = []
num_pert_pos_edges_in_between_list = []
num_total_pert_pos_edges_list = []

num_min_top_k_edges_in_protein_list = []
num_min_top_k_edges_in_ligand_list = []
num_min_top_k_edges_in_between_list = []
num_total_min_top_k_edges_list = []

fidelity_score_list = []
infidelity_score_list = []
trustworthiness_score_list = []

test_interaction_indices = []
num_test_interactions = SAMPLES_TO_EXPLAIN
probability_threshold = 0.75
test_interaction_names = []
test_interactions_affinities = []
test_interaction_names_affinities_dict = {}


for test_interaction_index in all_test_interaction_indices:
    model.eval()
    test_interaction = hold_out_data[test_interaction_index]

    edge_weight_to_pass = None
    if EDGE_WEIGHT:
        edge_weight_to_pass = test_interaction.edge_weight.to(device)

    batch = torch.zeros(test_interaction.x.shape[0], dtype=int, device=test_interaction.x.device)
    
    test_affinity_value = test_interaction.y.item()
    if AFFINITY_SET == "medium":
        if test_affinity_value < MEAN_LOWER_BOUND or test_affinity_value > MEAN_UPPER_BOUND:
            continue
    elif AFFINITY_SET == "low":
        if test_affinity_value >= LOW_BOUND:
            continue
    else:
        if test_affinity_value <= HIGH_BOUND:
            continue

    
    out = model(test_interaction.x.to(device), test_interaction.edge_index.to(device), batch=batch.to(device), edge_weight=edge_weight_to_pass)
    pred = out.item()
    if AFFINITY_SET == "low":
        if pred >= MEAN_LOWER_BOUND:
            continue
    elif AFFINITY_SET == "high":
        if pred <= MEAN_UPPER_BOUND:
            continue
    else:
        if pred < MEAN_LOWER_BOUND or pred > MEAN_UPPER_BOUND:
            continue
        
    test_interaction_indices.append(test_interaction_index)
    test_interaction_names.append(test_interaction.interaction_name)
    test_interactions_affinities.append(test_affinity_value)
    test_interaction_names_affinities_dict[test_interaction.interaction_name] = test_affinity_value
    
    if len(test_interaction_indices) == num_test_interactions:
        break


TARGET_CLASS = None

    
for index in tqdm(test_interaction_indices):
    model.eval()
    test_interaction = hold_out_data[index]
    print("\nInteraction: " + test_interaction.interaction_name)

    edge_weight_to_pass = None
    if EDGE_WEIGHT:
        edge_weight_to_pass = test_interaction.edge_weight.to(device)

    batch = torch.zeros(test_interaction.x.shape[0], dtype=int, device=test_interaction.x.device)
    
    
    out = model(test_interaction.x.to(device), test_interaction.edge_index.to(device), batch=batch.to(device), edge_weight=edge_weight_to_pass) #test_interaction.edge_weight.to(device)
    
    
    #explainability

    edgeshaper_explainer = Edgeshaper(model, test_interaction.x, test_interaction.edge_index, edge_weight = test_interaction.edge_weight, device = device)

    phi_edges = edgeshaper_explainer.explain(M = 100, target_class = TARGET_CLASS, deviation = None, seed = SEED) #deviation = 1e-3

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

    
    

    SAVE_PATH = SAVE_FOLDER + "/" + MODEL_NAME + "/" + AFFINITY_SET + " affinity" + "/" + test_interaction.interaction_name + "/"
    
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)


    with open(SAVE_PATH + test_interaction.interaction_name + "_statistics.txt", "w+") as f:
        f.write("Interaction name: " + test_interaction.interaction_name + "\n\n")
        f.write("Affinity: " + str(test_interaction.y.item()) + "\n")
        f.write("Predicted value: " + str(out.item()) + "\n\n")


        f.write("Shapley values for edges: \n\n")
        for i in range(len(phi_edges)):
            f.write("(" + str(test_interaction.edge_index[0][i].item()) + "," + str(test_interaction.edge_index[1][i].item()) + "): " + str(phi_edges[i]) + "\n")
