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

from src.utils import create_edge_index, PLIDataset, set_all_seeds, GCN, GraphSAGE, GAT, GIN, GINE, GC_GNN, save_model 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Working on device: ", device)

if __name__ == "__main__":
    with open("parameters.yml") as paramFile:  
        args = yaml.load(paramFile, Loader=yaml.FullLoader)

    DATA_PATH = args["trainer"]["DATA_PATH"]

    CLEAN_DATA = args["trainer"]["CLEAN_DATA"]
    MIN_AFFINITY = args["trainer"]["MIN_AFFINITY"]
    MAX_AFFINITY = args["trainer"]["MAX_AFFINITY"]
    NUM_CLASSES = 1 #set it up to 1 since we are facing a regression problem

    MEAN_LOWER_BOUND = args["trainer"]["MEAN_LOWER_BOUND"]
    MEAN_UPPER_BOUND = args["trainer"]["MEAN_UPPER_BOUND"]
    LOW_BOUND = args["trainer"]["LOW_BOUND"]
    HIGH_BOUND = args["trainer"]["HIGH_BOUND"]

    MODEL_NAME = args["trainer"]["GNN_MODEL"]

    GNN = GCN if MODEL_NAME == "GCN" else GraphSAGE if MODEL_NAME == "GraphSAGE" else GAT if MODEL_NAME == "GAT" else GIN if MODEL_NAME == "GIN" else GINE if MODEL_NAME == "GINE" else GC_GNN if MODEL_NAME == "GC_GNN" else None
    
    SAVE_BEST_MODEL = args["trainer"]["SAVE_BEST_MODEL"]
    MODEL_SAVE_FOLDER = args["trainer"]["MODEL_SAVE_FOLDER"]

    EDGE_WEIGHT = args["trainer"]["EDGE_WEIGHT"]
    SCALING = args["trainer"]["SCALING"]

    SEED = args["trainer"]["SEED"]
    HIDDEN_CHANNELS = args["trainer"]["HIDDEN_CHANNELS"]
    EPOCHS = args["trainer"]["EPOCHS"]
    NODE_FEATURES = args["trainer"]["NODE_FEATURES"] #if False, use dummy features (1)
    BATCH_SIZE = args["trainer"]["BATCH_SIZE"]
    LEARNING_RATE = float(args["trainer"]["LEARNING_RATE"])
    WEIGHT_DECAY = float(args["trainer"]["WEIGHT_DECAY"])
    
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

    # ### create torch dataset


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

    rng = np.random.default_rng(seed = SEED)
    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(core_set_data)
    rng.shuffle(hold_out_data)


    print("Number of samples after outlier removal: ", len(dataset))
    print("Number of training samples: ", len(train_data))
    print("Number of validation samples: ", len(val_data))
    print("Number of core set samples: ", len(core_set_data))
    print("Number of hold out samples: ", len(hold_out_data))

    core_set_hold_out_interactions = core_set_interactions + hold_out_interactions
    core_set_hold_out_data = [dataset[i] for i in range(len(dataset)) if dataset[i].interaction_name in core_set_hold_out_interactions]

    print("Number of test samples: ", len(core_set_hold_out_data))
    print("Number of test low affinity samples: ", len([sample for sample in core_set_hold_out_data if sample.y < LOW_BOUND]))
    print("Numbr of test medium affinity samples: ", len([sample for sample in core_set_hold_out_data if sample.y >= MEAN_LOWER_BOUND and sample.y <= MEAN_UPPER_BOUND]))
    print("Number of test high affinity samples: ", len([sample for sample in core_set_hold_out_data if sample.y > HIGH_BOUND]))


    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    core_set_loader = DataLoader(core_set_data, batch_size=BATCH_SIZE)
    hold_out_loader = DataLoader(hold_out_data, batch_size=BATCH_SIZE)


    # ### Train the network

    model = GNN(node_features_dim = dataset[0].x.shape[1], num_classes = NUM_CLASSES, hidden_channels=HIDDEN_CHANNELS).to(device)

    lr = LEARNING_RATE

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    criterion = torch.nn.MSELoss()
        
    epochs = EPOCHS


    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch, edge_weight = data.edge_weight)  # Perform a single forward pass.
            
            loss = torch.sqrt(criterion(torch.squeeze(out), data.y))  # Compute the loss.
        
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()

        sum_loss = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            
            out = model(data.x, data.edge_index, data.batch, edge_weight = data.edge_weight)  
            
            if  data.y.shape[0] == 1:
                loss = torch.sqrt(criterion(torch.squeeze(out, 1), data.y))
            else:
                loss = torch.sqrt(criterion(torch.squeeze(out), data.y)) * data.y.shape[0]
            sum_loss += loss.item()
            
        return sum_loss / len(loader.dataset) 



    best_epoch = 0

    best_val_loss = 100000
    for epoch in range(epochs):
        train()
        train_rmse = test(train_loader)
        val_rmse = test(val_loader)
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            best_epoch = epoch
            if SAVE_BEST_MODEL:
                save_model(model, MODEL_SAVE_FOLDER, model_name = MODEL_NAME + "_best")
            
        print(f'Epoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}')

    core_set_rmse = test(core_set_loader)    
    hold_out_set_rmse = test(hold_out_loader)

    if not SAVE_BEST_MODEL:
        print(f'Core set 2016 RMSE with latest model: {core_set_rmse:.4f}')
        print(f'Hold out set 2019 RMSE with latest model: {hold_out_set_rmse:.4f}')
        save_model(model, MODEL_SAVE_FOLDER, model_name = MODEL_NAME + "_latest", timestamp=True)

    print(f'Best model at epoch: {best_epoch:03d}')
    print("Best val loss: ", best_val_loss)


    if SAVE_BEST_MODEL:
        model = GNN(node_features_dim = dataset[0].x.shape[1], num_classes = NUM_CLASSES, hidden_channels=HIDDEN_CHANNELS).to(device)
        model.load_state_dict(torch.load("models/model_" + MODEL_NAME + "_best.ckpt"))
        model.to(device)

        core_set_rmse = test(core_set_loader)    
        print(f'Core set 2016 RMSE with best model: {core_set_rmse:.4f}')

        hold_out_set_rmse = test(hold_out_loader)    
        print(f'Hold out set 2019 RMSE with best model: {hold_out_set_rmse:.4f}')
 
        os.rename("models/model_" + MODEL_NAME + "_best.ckpt", "models/model_" + MODEL_NAME + "_best_" + str(best_epoch) + ".ckpt")

