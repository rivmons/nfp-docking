import torch
import torch.nn as nn
from features import \
    num_atom_features, \
    num_bond_features
import numpy as np
from util import buildFeats
from torch.utils.data import Dataset
import torch.nn.functional as F
import os

# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")
print(f'num interop threads: {torch.get_num_interop_threads()}, num intraop threads: {torch.get_num_threads()}') 

class dockingDataset(Dataset):
    def __init__(self, train, labels, maxa=70, maxd=6):
        # self.train = (zid, smile), self.label = (bin label)
        self.train = train
        self.labels = torch.from_numpy(np.array(labels)).float()
        self.maxA = maxa
        self.maxD = maxd
        self.a, self.b, self.e = buildFeats([x[1] for x in self.train], self.maxD, self.maxA)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.a[idx], self.b[idx], self.e[idx], (self.labels[idx], self.train[idx][0])

class GraphLookup(nn.Module):
    def __init__(self):
        super().__init__()
        self.to(device)

    def temporal_padding(self, x, paddings=(1, 0), pad_value=0):
        if not isinstance(paddings, (tuple, list, np.ndarray)):
            paddings = (paddings, paddings)
        output = torch.zeros(x.size(0), x.size(1) + sum(paddings), x.size(2), device=device)
        output[:, :paddings[0], :] = pad_value
        output[:, paddings[1]:, :] = pad_value
        output[:, paddings[0]: paddings[0]+x.size(1), :] = x
        return output
    
    def lookup_neighbors(self, atoms, edges, maskvalue=0, include_self=False):
        masked_edges = edges + 1
        masked_atoms = self.temporal_padding(atoms, (1, 0), pad_value=maskvalue)

        batch_n, lookup_size, n_atom_features = masked_atoms.size()
        _, max_atoms, max_degree = masked_edges.size()

        expanded_atoms = masked_atoms.unsqueeze(2).expand(batch_n, lookup_size, max_degree, n_atom_features)
        expanded_edges = masked_edges.unsqueeze(3).expand(batch_n, max_atoms, max_degree, n_atom_features)
        output = torch.gather(expanded_atoms, 1, expanded_edges)

        if include_self:
            return torch.cat([(atoms.view(batch_n, max_atoms, 1, n_atom_features)), output], dim=2)
        return output

    def forward(self, atoms, edges, maskvalue=0, include_self=True):
        atoms, edges = atoms.to(device), edges.to(device)
        return self.lookup_neighbors(atoms, edges, maskvalue, include_self)

class nfpConv(nn.Module):
    def __init__(self, ishape, oshape):
        super(nfpConv, self).__init__()
        self.ishape = ishape
        self.oshape = oshape

        self.w = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty((self.ishape, self.oshape))), requires_grad=True).to(device)
        self.b = nn.Parameter(torch.nn.init.constant_(torch.empty((1, self.oshape)), 0.01), requires_grad=True).to(device)

        # self.w = torch.nn.init.xavier_normal_(self.w)
        # self.b = torch.nn.init.constant_(self.b, 0.01)
        
        self.degArr = nn.ParameterList([nn.Parameter(torch.nn.init.xavier_normal_(torch.empty((self.ishape + 6, self.oshape))), requires_grad=True).to(device) for _ in range(6)])
       
        self.graphLookup = GraphLookup()  
        self.to(device)

    # def initDegreeWeights(self):
    #     for _ in range(6):
    #         dw = torch.empty((self.ishape + num_bond_features(), self.oshape), device=device)
    #         dw = nn.Parameter(dw)
    #         dw = nn.init.xavier_normal_(dw)
    #         self.degArr.append(dw)

    def forward(self, input):
        atoms, bonds, edges = input
        atoms, bonds, edges = atoms.to(device), bonds.to(device), edges.to(device)
        atom_degrees = (edges != -1).sum(-1, keepdim=True)
        neighbor_atom_features = self.graphLookup(atoms, edges, include_self=True)
        summed_atom_features = neighbor_atom_features.sum(-2)
        summed_bond_features = bonds.sum(-2)
        summed_features = torch.cat([summed_atom_features, summed_bond_features], dim=-1)

        new_features = None
        for degree in range(6):
            atom_masks_this_degree = (atom_degrees == degree).float()
            new_unmasked_features = F.sigmoid(torch.matmul(summed_features, self.degArr[degree]) + self.b)
            # atom_activations.append(new_unmasked_features)
            new_masked_features = new_unmasked_features * atom_masks_this_degree
            new_features = new_masked_features if degree == 0 else new_features + new_masked_features

        return new_features


class nfpOutput(nn.Module):
    def __init__(self, layer, fpl):
        super(nfpOutput, self).__init__()
        self.fpl = fpl
        self.layer = layer

        w = torch.empty((self.layer + num_bond_features(), self.fpl), device=device)
        self.w = nn.Parameter(w)
        b = torch.empty((1, self.fpl), device=device)
        self.b = nn.Parameter(b)

        # self.bound = 1/np.sqrt(layer)
        torch.nn.init.xavier_normal_(self.w)
        torch.nn.init.constant_(self.b, 0.01)
        self.to(device)

    def forward(self, a, b, e):
        a, b, e = a.to(device), b.to(device), e.to(device)
        atom_degrees = (e != -1).sum(-1, keepdim=True)
        general_atom_mask = (atom_degrees != 0).float()
        summed_bond_features = b.sum(-2)
        summed_features = torch.cat([a, summed_bond_features], dim=-1)
        fingerprint_out_unmasked = F.sigmoid(torch.matmul(summed_features, self.w) + self.b)
        fingerprint_out_masked = fingerprint_out_unmasked * general_atom_mask

        return fingerprint_out_masked.sum(dim=-2), fingerprint_out_masked
    
class GraphPool(nn.Module):
    def __init__(self):
        super(GraphPool, self).__init__()
        self.graphLookup = GraphLookup()
        self.to(device)

    def forward(self, atoms, edges):
        atoms, edges = atoms.to(device), edges.to(device)
        neighbor_atom_features = self.graphLookup(atoms, edges, maskvalue=-np.inf, include_self=True)
        max_features = neighbor_atom_features.max(dim=2)[0]
        atom_degrees = (edges != -1).sum(dim=-1, keepdim=True)
        general_atom_mask = (atom_degrees != 0).float()
        return max_features * general_atom_mask


class nfpDocking(nn.Module):
    def __init__(self, layers, return_activations, fpl=32, hf=20):
        super(nfpDocking, self).__init__()
        self.layers = layers
        self.fpl = fpl
        # self.hiddenFeat = hf
        self.throughShape = list(zip(layers[:-1], layers[1:]))
        self.layersArr, self.outputArr = self.init_layers()
        self.op = nfpOutput(self.layers[-1], self.fpl)
        self.pool = GraphPool()
        self.return_activations = return_activations
        self.atom_activations = None
        if self.return_activations:
            self.atom_activations = []
        self.to(device)

    def init_layers(self):
        layersArr, outputArr = [], []
        for idx, (i, o) in enumerate(self.throughShape):
            outputArr.append(nfpOutput(self.layers[idx], self.fpl))
            layersArr.append(nfpConv(i, o))
        outputArr.append(nfpOutput(self.layers[-1], self.fpl))
        return nn.ModuleList(layersArr), nn.ModuleList(outputArr)
            
    
    def forward(self, input):
        self.atom_activations = []
        a, b, e = input
        a, b, e = a.to(device), b.to(device), e.to(device)
        ffp = torch.zeros(a.shape[0], self.fpl).to(device)
        for i in range(len(self.layers[1:])):
            # if self.return_activations:
            #     a, atom_act = self.layersArr[i]((a, b, e))
            #     self.atom_activations.append(atom_act)
            # else:
            #     a = self.layersArr[i]((a, b, e))
            # a = self.pool(a, e)
            lfp, aact = self.outputArr[i](a, b, e)
            ffp += lfp
            if self.return_activations: self.atom_activations.append(aact)
            a = self.layersArr[i]((a, b, e))
            a = self.pool(a, e)
        ffp, aact = self.op(a, b, e)
        if self.return_activations: self.atom_activations.append(aact)
        return ffp
    
class dockingANN(nn.Module):
    def __init__(self, fpl, ba, layers, dropout):
        super(dockingANN, self).__init__()
        self.inputSize = fpl
        self.ba = ba
        self.arch = list(zip(ba[:-1], ba[1:]))
        self.layers = layers
        self.dropout = dropout

        self.ann = nn.Sequential()
        self.buildModel()
        self.to(device)

    def buildModel(self):
        for j, (i, o) in enumerate(self.arch):
            self.ann.add_module(f'linear {j}', nn.Linear(i, o))
            # b = 0.01 if j != len(self.arch) - 1 else np.log([self.pos/self.neg])[0]
            self.ann[-1].bias = torch.nn.init.constant_(torch.nn.Parameter(torch.empty(o, device=device)), 0.01)
            if o != 1:
                self.ann.add_module(f'batch norm {j}', nn.BatchNorm1d(o))
                self.ann.add_module(f'relu act {j}', nn.ReLU())
                self.ann.add_module(f'dropout {j}', nn.Dropout(self.dropout))
            # else: self.ann.add_module(f'relu act {j}', nn.ReLU())
        # self.ann.add_module(f'output', nn.Sigmoid()) # * for bcewithlogitsloss

    def forward(self, input):
        # input = torch.tensor(input, device=device)
        return self.ann(input)
    
class dockingProtocol(nn.Module):
    def __init__(self, params):
        super(dockingProtocol, self).__init__()
        self.model = nn.Sequential(
            nfpDocking(
                layers=params["conv"]["layers"],
                fpl=params["fpl"],
                return_activations=params["conv"]["activations"]
             ),
            dockingANN(
                fpl=params["fpl"],
                ba=params["ann"]["ba"],
                dropout=params["ann"]["dropout"],
                layers=params["ann"]["layers"]
            )
        )
        self.return_activations = params["conv"]["activations"]
        self.to(device)

    def forward(self, input):
        if self.return_activations:
            return torch.squeeze(self.model(input)), self.model[0].atom_activations # torch.squeeze
        return torch.squeeze(self.model(input))

class dockingDataset(Dataset):
    def __init__(self, train, labels, maxa=70, maxd=6, name='unknown'):
        # self.train = (zid, smile), self.label = (bin label)
        self.train = train
        self.labels = torch.from_numpy(np.array(labels)).float()
        self.maxA = maxa
        self.maxD = maxd
        self.a, self.b, self.e = buildFeats([x[1] for x in self.train], self.maxD, self.maxA, name)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return self.a[idx], self.b[idx], self.e[idx], (self.labels[idx], self.train[idx][0])