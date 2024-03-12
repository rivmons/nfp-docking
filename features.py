# atom_features: encode atom symbol, num bonded atoms, num hydrogens, implicit valence, if aromatic

import numpy as np
from rdkit import Chem

def encoding(feat, featArray):
    if feat not in featArray:
        feat = featArray[0]
    return list(map(lambda f: int(feat == f), featArray))

def getAtomFeatures(atom):
    symbolArray = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                    'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                    'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                    'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                    'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    degArray = [x for x in range(6)]
    return np.array([
        *encoding(atom.GetSymbol(), symbolArray),
        *encoding(atom.GetDegree(), degArray),
        *encoding(atom.GetTotalNumHs(), degArray[:-1]),
        *encoding(atom.GetImplicitValence(), degArray),
        int(atom.GetIsAromatic())
    ])

def getBondFeatures(bond):
    bondType = bond.GetBondType()
    return np.array(list(map(int, [
        bondType == Chem.rdchem.BondType.SINGLE,
        bondType == Chem.rdchem.BondType.DOUBLE,
        bondType == Chem.rdchem.BondType.TRIPLE,
        bondType == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ])))
    
def num_atom_features():
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(getAtomFeatures(a))

def num_bond_features():
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(getBondFeatures(simple_mol.GetBonds()[0]))

