### Module implementing prediction phase for additional experiments ###
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Working on device: ", device)

with open("parameters.yml") as paramFile:  
        args = yaml.load(paramFile, Loader=yaml.FullLoader)

DATA_PATH = args["affinity_shifting"]["DATA_PATH"]

CLEAN_DATA = args["affinity_shifting"]["CLEAN_DATA"]
MIN_AFFINITY = args["affinity_shifting"]["MIN_AFFINITY"]
MAX_AFFINITY = args["affinity_shifting"]["MAX_AFFINITY"]
NUM_CLASSES = 1 #set it up to 1 since we are facing a regression problem

MEAN_LOWER_BOUND = args["affinity_shifting"]["MEAN_LOWER_BOUND"]
MEAN_UPPER_BOUND = args["affinity_shifting"]["MEAN_UPPER_BOUND"]
LOW_BOUND = args["affinity_shifting"]["LOW_BOUND"]
HIGH_BOUND = args["affinity_shifting"]["HIGH_BOUND"]

MODEL_NAME = args["affinity_shifting"]["GNN_MODEL"]
GNN = GCN if MODEL_NAME == "GCN" else GraphSAGE if MODEL_NAME == "GraphSAGE" else GAT if MODEL_NAME == "GAT" else GIN if MODEL_NAME == "GIN" else GINE if MODEL_NAME == "GINE" else GC_GNN if MODEL_NAME == "GC_GNN" else None
MODEL_PATH = args["affinity_shifting"]["MODEL_PATH"]

print("Using model: ", MODEL_NAME)

EDGE_WEIGHT = args["affinity_shifting"]["EDGE_WEIGHT"]
SCALING = args["affinity_shifting"]["SCALING"]
BATCH_SIZE = args["affinity_shifting"]["BATCH_SIZE"]
LEARNING_RATE = float(args["affinity_shifting"]["LEARNING_RATE"])
WEIGHT_DECAY = float(args["affinity_shifting"]["WEIGHT_DECAY"])

SEED = args["affinity_shifting"]["SEED"]
HIDDEN_CHANNELS = args["affinity_shifting"]["HIDDEN_CHANNELS"]
EPOCHS = args["affinity_shifting"]["EPOCHS"]
NODE_FEATURES = args["affinity_shifting"]["NODE_FEATURES"] #if False, use dummy features (1)

#parameters for additional experiments for revision

AFFINITY_SHIFT = float(args["affinity_shifting"]["AFFINITY_SHIFT"])

############################

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
        dirs = sorted(dirs, key = str)
        
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
                    dataset_dict[interaction_name]["y"] = torch.FloatTensor([interaction_affinities[interaction_name]["affinity"] + AFFINITY_SHIFT]) #add AFFINITY_SHIFT to affinity to check prediction behavior

        
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
print(f'Core set 2016 RMSE with loaded model (affinities increased by {AFFINITY_SHIFT}): {core_set_rmse:.4f}')

hold_out_set_rmse = test(hold_out_loader)    
print(f'Hold out set 2019 RMSE with loaded model (affinities increased by {AFFINITY_SHIFT}): {hold_out_set_rmse:.4f}')