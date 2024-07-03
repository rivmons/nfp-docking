import torch
import torch.nn as nn
import argparse
import pandas as pd
import numpy as np
from math import ceil
from features import \
    num_atom_features, \
    num_bond_features
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import sys
from networkP import dockingProtocol, GraphLookup
from util import buildFeats, dockingDataset, labelsToDF, find_item_with_keywords
import time
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from scipy.stats import linregress


def group_most_associated_w_fp_feature(fp_i, fp_size, degree_activations):
    """Produce a dict of {Molecule: group (atom-radii) that most activates a specific fingerprint feature}'s"""
    most_associated_activations = {}

    # for each molecule, find the degree-atom pair with the highest value at the fingerprint feature index
    for degree, activations_dict in degree_activations.items():
        for zinc_id, tensor in activations_dict.items():
            assert fp_size == tensor.shape[1], f"fp_size {fp_size} does not match hidden layer feature size {tensor.shape[1]}"
            assert fp_i < tensor.shape[1], f"Index {fp_i} is out of bounds for tensor shape {tensor.shape[1]}"
            
            assoc_values = tensor[:,fp_i]
            max_index = torch.argmax(assoc_values).item()
            max_activations = tensor[max_index, :]

            if zinc_id not in most_associated_activations:
                most_associated_activations[zinc_id] = (degree, max_activations, max_index)
            else:
                old_max = most_associated_activations[zinc_id][1][fp_i]
                new_max = max_activations[fp_i]
                if new_max>old_max:
                    most_associated_activations[zinc_id] = (degree, max_activations, max_index)

    return most_associated_activations


def get_smile_from_zinc_id(zinc_id, reference):
    ## note to self: when construct new synthetic mols, will need to handle z_id/smile labeling; maybe add to master-data-dock & give me names "SYNTHETIC1020"    
    try:
        smile = smileData.loc[zinc_id, 'smile']
        return smile
    except KeyError:
        return f"ZINC ID {zinc_id} not found."


def get_atom_neighborhood(smile, center_atom_i, max_degree):
    # max deg = 0 (just central), = 1 (central+first neighbors), ...
    a,_,_ = buildFeats(smile)
    atom_neighborhood = [center_atom_i]

    for neighbor_degree in range(max_degree):
        for atom in list(atom_neighborhood): # iter over a copy to avoid reading neighbors as they're added
            neighbors = e[0, atom, :]
            for neighbor_slot in neighbors:
                neighbor_i = neighbor_slot.item()
                if neighbor_i != -1 and neighbor_i not in atom_neighborhood:  # -1 == neighbor doesn't exist
                    atom_neighborhood.append(neighbor_i)

    return atom_neighborhood


def draw_molecule_with_highlights(filename, smiles, highlight_atoms):
    figsize = (300, 300)
    highlight_color = (40.0/255.0, 200.0/255.0, 80.0/255.0) 

    drawoptions = DrawingOptions()
    drawoptions.selectColor = highlight_color
    drawoptions.elemDict = {}
    drawoptions.bgColor=None

    mol = Chem.MolFromSmiles(smiles)
    fig = Draw.MolToMPL(mol, highlightAtoms=highlight_atoms, size=figsize, options=drawoptions,fitImage=False)

    fig.gca().set_axis_off()
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def setup_dataset(input_data, name, reference, input_only=False):
    # input zIDs
    allData = labelsToDF(f'./data/dock_{input_data}.txt')
    allData.set_index('zinc_id', inplace=True)
    allData = pd.merge(allData, reference, on='zinc_id')

    xData = [(index, row['smile']) for index, row in allData.iterrows()] # (zinc_id, smile)
    if input_only:
        yData = [0] * len(xData)
    else: 
        yData = allData['labels'].values.tolist()

    dataset = dockingDataset(train=xData, 
                            labels=yData,
                            name=name)
    return dataset



def find_most_predictive_features(loaded_model, orig_data, reference):
    """Compare each fp feature vs labels, get R^2, return most anti/correlated features"""

    # calculate fingerprints for original dataset
    dataset = setup_dataset(input_data=orig_data, name="Calc. FP's", reference=smileData)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    fp_dict = {}
    for batch, (a, b, e, (y, zID)) in enumerate(dataloader):
            at, bo, ed, Y = a.to(device), b.to(device), e.to(device), y.to(device)
            _, fps = loaded_model((at, bo, ed), return_fp=True)
            print("FPS:", fps.shape)
            fps = fps.detach().numpy()

            for i, z_id in enumerate(zID):
                fp_dict[z_id] = fps[i]

    # calculate (labels, feature) correlations
    orig_data_paths = find_item_with_keywords(search_dir="./data", keywords=[orig_data], file=True)
    orig_data_path = min(orig_data_paths, key=len)
    origData = labelsToDF(orig_data_path)
    origData.set_index('zinc_id', inplace=True)
    np_index = origData.index.values
    np_array = origData.values

    m = np_index.shape[0]
    first_fp = next(iter(fp_dict.values()))
    fp_len = first_fp.shape[0]
    n = fp_len
    fp_arr = np.zeros((m, n))

    for i, z_id in enumerate(np_index):
        if z_id not in fp_dict:
            print(f"Fingerprint not found for molecule {i} in original dataset.")
            continue
        fp = fp_dict[z_id] # makes sure z_id<->z_id,fp aligned
        fp_arr[i,:] = fp
    merged_arr = np.concatenate([np_array, fp_arr], axis=1)

    corr_list = []
    for i, fp_feature in enumerate(merged_arr[:, 1:].T):  
        labels = merged_arr[:,0]
        slope, intercept, r_value, p_value, std_err = linregress(labels, fp_feature)
        r_squared = r_value ** 2
        corr_list.append({'Feature #': i, 'R': r_value, 'R^2': r_squared, 'P': p_value})

    max_R_feat = max(corr_list, key=lambda x: x['R'])
    max_Rsquared_feat = max(corr_list, key=lambda x: x['R^2'])
    min_R_feat = min(corr_list, key=lambda x: x['R'])

    print(f"---- For {orig_data} model ----\nMost corr:{max_R_feat}\nMost anticorr:{min_R_feat}")
    return max_R_feat, min_R_feat


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

input = "acease_pruned"

# reference SMILEs/zID
smileData = pd.read_csv('./data/smilesDS.smi', delimiter=' ')
smileData.columns = ['smile', 'zinc_id']
smileData.set_index('zinc_id', inplace=True)

dataset = setup_dataset(input_data=input, name="Get conv. activations", reference=smileData, input_only=True)
dataloader = DataLoader(dataset, batch_size=12, shuffle=False)

# import model
model_path = "/data/users/vantilme1803/nfp-docking/src/trainingJobs/acease_pruned_model_1.pt"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model = dockingProtocol(params=checkpoint['params'])
model.load_state_dict(checkpoint['model_state_dict'])

fp_dict = {}
degree_activations = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
# feed SMILEs into model, get out activations
for batch, (a, b, e, (y, zID)) in enumerate(dataloader):
        at, bo, ed, Y = a.to(device), b.to(device), e.to(device), y.to(device)
        activs, fps, preds = model((at, bo, ed), return_conv_activs=True, return_fp=True)
        fps = fps.detach().numpy()

        for i, z_id in enumerate(zID):
            fp_dict[z_id] = fps[i]

        for degreeTuple in activs:
            degree = degreeTuple[0]
            vec = degreeTuple[1]
            assert degree in degree_activations, f"Unexpected degree: {degree}"

            activation_dict = degree_activations[degree]
            for i, z_id in enumerate(zID):
                activation_dict[z_id] = vec[i,:,:]

best_feat, worst_feat = find_most_predictive_features(model, checkpoint['dataset'], smileData)
first_fp = next(iter(fp_dict.values()))
fp_len = first_fp.shape[0]

best_subgraphs_dict = group_most_associated_w_fp_feature(best_feat['Feature #'],fp_len,degree_activations)

for i, (zinc_id, atomTuple) in enumerate(best_subgraphs_dict.items()):
    smile = get_smile_from_zinc_id(zinc_id, smileData)
    degree = atomTuple[0]
    atom_index = atomTuple[2]
    atom_neighborhood = get_atom_neighborhood([smile], atom_index, degree)
    if i==0 or i==1 or i==2:
        print(f"Molecule {i}:", zinc_id, "- best atoms:", atom_neighborhood)
        draw_molecule_with_highlights(f"{zinc_id}.png", smile, atom_neighborhood) # note both are RDKit ordering, indices align





