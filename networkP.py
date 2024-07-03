import torch
import torch.nn as nn
from features import \
    num_atom_features, \
    num_bond_features
import numpy as np
from util import buildFeats
from util import dockingDataset
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

    def forward(self, input, return_conv_activs=False):
        atoms, bonds, edges = input
        atoms, bonds, edges = atoms.to(device), bonds.to(device), edges.to(device)
        atom_degrees = (edges != -1).sum(-1, keepdim=True)
        neighbor_atom_features = self.graphLookup(atoms, edges, include_self=True)
        summed_atom_features = neighbor_atom_features.sum(-2)
        summed_bond_features = bonds.sum(-2)
        summed_features = torch.cat([summed_atom_features, summed_bond_features], dim=-1)

        new_features = None
        stored_activations = []
        for degree in range(6): # no atom has >5 bonds
            atom_masks_this_degree = (atom_degrees == degree).float()
            new_unmasked_features = F.relu(torch.matmul(summed_features, self.degArr[degree]) + self.b)
            new_masked_features = new_unmasked_features * atom_masks_this_degree
            if return_conv_activs:
                stored_activations.append((degree, new_masked_features))
            new_features = new_masked_features if degree == 0 else new_features + new_masked_features

        if stored_activations:
            return stored_activations, new_features
        else: return new_features


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
        fingerprint_out_unmasked = torch.sigmoid(torch.matmul(summed_features, self.w) + self.b)
        fingerprint_out_masked = fingerprint_out_unmasked * general_atom_mask

        return fingerprint_out_masked.sum(dim=-2)
    
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
    def __init__(self, layers, fpl=32, hf=32):
        super(nfpDocking, self).__init__()
        self.layers = layers
        self.fpl = fpl
        # self.hiddenFeat = hf
        self.throughShape = list(zip(layers[:-1], layers[1:]))
        self.layersArr, self.outputArr = self.init_layers()
        self.op = nfpOutput(self.layers[-1], self.fpl)
        self.pool = GraphPool()
        self.to(device)

    def init_layers(self):
        layersArr, outputArr = [], []
        for idx, (i, o) in enumerate(self.throughShape):
            outputArr.append(nfpOutput(self.layers[idx], self.fpl))
            layersArr.append(nfpConv(i, o))
        outputArr.append(nfpOutput(self.layers[-1], self.fpl))
        return nn.ModuleList(layersArr), nn.ModuleList(outputArr)
            
    
    def forward(self, input, return_conv_activs=False):
        a, b, e = input
        a, b, e = a.to(device), b.to(device), e.to(device)
        lay_count = len(self.layers[1:])
        for i in range(lay_count):
            if return_conv_activs and i == lay_count-1: # if last store activations
                activations, a = self.layersArr[i]((a, b, e), return_conv_activs=True)
            else:
                a = self.layersArr[i]((a, b, e)) # calls nfpConv layer on inputs
            a = self.pool(a, e)
        
        fp = self.op(a, b, e)
        if return_conv_activs:
            return activations, fp
        else: return fp
    
class ActivationCapture(nn.Module):
    def __init__(self):
        super(ActivationCapture, self).__init__()
        self.activations = None

    def forward(self, x):
        self.activations = x.clone()
        return x

class dockingANN(nn.Module):
    def __init__(self, fpl, ba, layers, dropout):
        super(dockingANN, self).__init__()
        self.inputSize = fpl
        self.ba = ba
        self.arch = list(zip(ba[:-1], ba[1:]))
        self.layers = layers
        self.dropout = dropout

        self.ann = nn.Sequential()
        self.fp_activs = []
        self.buildModel()
        self.to(device)
    
    def buildModel(self):
        for j, (i, o) in enumerate(self.arch):
            if j == 0:
                self.ann.add_module(f'fingerprint_capture', ActivationCapture())
            self.ann.add_module(f'linear {j}', nn.Linear(i, o))
            # b = 0.01 if j != len(self.arch) - 1 else np.log([self.pos/self.neg])[0]
            self.ann[-1].bias = torch.nn.init.constant_(torch.nn.Parameter(torch.empty(o, device=device)), 0.01)
            self.ann.add_module(f'relu act {j}', nn.ReLU())
            self.ann.add_module(f'batch norm {j}', nn.BatchNorm1d(o))
            self.ann.add_module(f'dropout {j}', nn.Dropout(self.dropout))
        self.ann.add_module(f'output', nn.Sigmoid())

    def forward(self, input, return_fp=False):
        # input = torch.tensor(input, device=device)
        output_from_sequential = self.ann(input)
        activations=None
        for module in self.ann.children(): # return first layer FP
            if isinstance(module, ActivationCapture) and return_fp:
                activations = module.activations

        if activations is not None: 
            return output_from_sequential, activations
        return output_from_sequential

    
class dockingProtocol(nn.Module):
    def __init__(self, params):
        super(dockingProtocol, self).__init__()
        self.model = nn.Sequential(
            nfpDocking(
                layers=params["conv"]["layers"],
                fpl=params["fpl"]
            ),
            dockingANN(
                fpl=params["fpl"],
                ba=params["ann"]["ba"],
                dropout=params["ann"]["dropout"],
                layers=params["ann"]["layers"]
            )
        )
        self.to(device)

    def forward(self, input, return_conv_activs=False, return_fp=False):
        if return_conv_activs and return_fp:
            conv_activs, fp_input = self.model[0](input, return_conv_activs=True) # run conv on inputs
            pred, fp_activs = self.model[1](fp_input, return_fp=True) # run linears on conv-outputs
            return conv_activs, fp_activs, pred
        elif return_conv_activs:
            conv_activs, fp_input = self.model[0](input, return_conv_activs=True)
            return conv_activs, torch.squeeze(self.model[1](fp_input))
        elif return_fp:
            fp_input = self.model[0](input)
            return self.model[1](fp_input, return_fp=True)
        else:
            fp_input = self.model[0](input)
            return torch.squeeze(self.model[1](fp_input))
    
    def save(self, params, dataset, outpath):
        torch.save({
            'model_state_dict': self.state_dict(),
            'params': params,
            'dataset': dataset
        }, outpath)