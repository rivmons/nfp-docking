import torch as T
import numpy as np
import pandas as pd
from features import num_atom_features, num_bond_features, getAtomFeatures, getBondFeatures
from tqdm import tqdm
from rdkit import Chem
from typing import List
import sys
import os
from torch.utils.data import Dataset, DataLoader


def padDim(arr, size, dim, val=0, padR=True):
    padded = [(0, 0)] * len(arr.shape)
    padded[dim] = (0, size - arr.shape[dim]) if padR else (size - arr.shape[dim], 0)
    return np.pad(arr, pad_width=padded, mode='constant', constant_values=val)

def buildFeats(smiles, maxDeg=5, maxAtom=70, ds='unknown'):
    n = len(smiles)
    nAF = num_atom_features()
    nBF = num_bond_features()
    atoms = np.zeros((n, maxAtom, nAF))
    bonds = np.zeros((n, maxAtom, maxDeg, nBF))
    edges = -np.ones((n, maxAtom, maxDeg), dtype=int)

    for molIdx, smile in enumerate(smiles):
        molecule = Chem.MolFromSmiles(smile)
        molAtoms = molecule.GetAtoms()
        molBonds = molecule.GetBonds()
        idxMap = {}
        connMat = [[] for i in range(len(molAtoms))]

        if len(molAtoms) > atoms.shape[1]:
            atoms = padDim(atoms, len(molAtoms), axis=1)
            bonds = padDim(bonds, len(molAtoms), axis=1)
            edges = padDim(edges, len(molAtoms), axis=1, val=-1)
        
        for atomIdx, atom in enumerate(molAtoms):
            atoms[molIdx, atomIdx, : nAF] = getAtomFeatures(atom)
            idxMap[atom.GetIdx()] = atomIdx

        for bond in molBonds:
            atom1Idx = idxMap[bond.GetBeginAtom().GetIdx()]
            atom2Idx = idxMap[bond.GetEndAtom().GetIdx()]
            atom1Neighbor = len(connMat[atom1Idx])
            atom2Neighbor = len(connMat[atom2Idx])

            if max(atom1Neighbor, atom2Neighbor) + 1 > bonds.shape[2]:
                bonds = padDim(bonds, max(atom1Neighbor, atom2Neighbor) + 1, axis=2)
                edges = padDim(edges, max(atom1Neighbor, atom2Neighbor) + 1, axis=2, val=-1)
            
            bondFeat = np.array(getBondFeatures(bond), dtype=int)
            bonds[molIdx, atom1Idx, atom1Neighbor, :] = bondFeat
            bonds[molIdx, atom2Idx, atom2Neighbor, :] = bondFeat

            connMat[atom1Idx].append(atom2Idx)
            connMat[atom2Idx].append(atom1Idx)
        
        for atom1Idx, ngb in enumerate(connMat):
            d = len(ngb)
            edges[molIdx, atom1Idx, : d] = ngb

    return T.from_numpy(atoms).float(), T.from_numpy(bonds).float(), T.from_numpy(edges).long()

def find_item_with_keywords(search_dir, keywords: List[str], dir=False, file=False):
    items = os.listdir(search_dir)
    if dir == True:
        filtered = [item for item in items if os.path.isdir(os.path.join(search_dir, item))]
    elif file == True:
        filtered = [item for item in items if os.path.isfile(os.path.join(search_dir, item))]

    accepted_items = []
    for obj in filtered:
        if all(keyword in obj for keyword in keywords):
            full_path = os.path.join(search_dir, obj)
            accepted_items.append(full_path)

    return accepted_items

class dockingDataset(Dataset):
    def __init__(self, train, labels, 
    maxa=70, maxd=6, # 6 bond hard cap; 70 atom soft limit
    name='unknown'):
        # self.train = (zid, smile), self.label = (bin label)
        self.train = train
        self.labels = T.from_numpy(np.array(labels)).float()
        self.maxA = maxa
        self.maxD = maxd
        smiles = [x[1] for x in self.train]
        self.a, self.b, self.e = buildFeats(smiles, self.maxD, self.maxA, name)
        self.zinc_ids = [x[0] for x in self.train]
        self.smiles = [x[1] for x in self.train]

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return self.a[idx], self.b[idx], self.e[idx], (self.labels[idx], self.zinc_ids[idx])

def check_first_row_labels(file):
    """Check if first row of an input tab-delin .txt is labels"""
    first_line = file.readline().strip().lstrip('\ufeff')
    values = first_line.split('\t')
    for value in values:
        # strip, check if the remaining is entirely numeric
        stripped = value.translate(value.maketrans('', '', '. -'))
        if stripped.isdigit():
            return None
    return values

def is_consecutive_list(list):
    for i,elem in enumerate(list[:5]):
        if int(list[i+1]) != int(elem)+1:
            return False
    return True

def remove_empty_strings_from_list(lst):
    return list(filter(None, lst))
    

def labelsToDF(fname):
    arr = []
    with open(fname) as f:
        labels = check_first_row_labels(f)
        if labels is not None:
            next(f)
        
        for line in f.readlines():
            data = line.strip().split('\t')
            data = [item for item in data if item.strip() != '']
            arr.append(data)
    
    if labels is not None:
        df = pd.DataFrame(arr, columns=labels)
        df['labels'] = df['labels'].astype(float)
    else:
        ## assumes it is a docking .txt file
        index_list = [arr[i][0][0] for i in range(10)]
        includes_ind = is_consecutive_list(index_list)
        if includes_ind:
            for i in range(len(arr)):
                arr[i][0] = float(arr[i][0].split()[1]) #strip indices
        
        df = pd.DataFrame(arr, columns=['labels','zinc_id'])
    
    return df