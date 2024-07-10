import torch as T
import numpy as np
from features import num_atom_features, num_bond_features, getAtomFeatures, getBondFeatures
from tqdm import tqdm
from rdkit import Chem


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

    print(f'building tensors for {ds} dataset')
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