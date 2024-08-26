import pickle
import torch
from networkE import dockingProtocol, dockingDataset, nfpDocking
from torch.utils.data import DataLoader
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from features import getAtomFeatures, num_atom_features
import os
import io
from PIL import Image
import glob
import statistics
import argparse
from util import buildFeats

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('-time', '--t', default=False, required=False)
parser.add_argument('-substructure', '--s', default=False, required=False)
parser.add_argument('-permutation', '--p', default=False, required=False)

dude = True

args = parser.parse_args()
tm = bool(args.t)
substr = bool(args.s)
permutation = bool(args.p)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

path = './esr1/'
mn = 1

modelp = None
with open(f'{path}res/model{mn}/modelparams.pkl', 'rb') as f:
    modelp = pickle.load(f)

testf = open(f'{path}res/model{mn}/testset.txt', 'r')
testd = [i.strip().split(",") for i in testf.readlines()[1:]]

smilesdict = None
if not dude:
    smilesf = open('./data/smilesDS.smi', 'r')
    smilesd = [i.strip().split(" ") for i in smilesf.readlines()]
    smilesdict = {
        i[1]: i[0]
        for i in smilesd
    }
else:
    protein_path = './dude/' + path.split('/')[1]

    activef = open(f'{protein_path}/actives_final.ism', 'r')
    decoyf = open(f'{protein_path}/decoys_final.ism', 'r')

    actives_d = {
        line[1] : line[0]
        for line in [l.strip().split(" ") for l in activef.readlines()]
    }
    decoys_d = {
        line[1] : line[0]
        for line in [l.strip().split(" ") for l in decoyf.readlines()]
    }
    
    smilesdict = actives_d
    smilesdict.update(decoys_d)

for i, mol in enumerate(testd):
    testd[i].append(smilesdict[mol[0]])

# [zid, smile], 0/1
cf = -10
xtest = [[i[0], i[3]] for i in testd]
ytest = [int(float(i[2]) < cf) for i in testd]

modelp["conv"]["activations"] = True
model = dockingProtocol(modelp).to(device=device)
model.load_state_dict(torch.load(f'{path}models/model{mn}.pth', map_location=torch.device('cpu')), strict=False)
model.eval()

testds = dockingDataset(train=xtest,
                        labels=ytest,
                        name='test')
testdl = DataLoader(testds, batch_size=512, shuffle=False)
print('done building tensors')

activations = np.empty((len(modelp["conv"]["layers"]), len(xtest), 70, modelp["fpl"]))
activations_zid = []
mol = 0
for (a, b, e, (y, zid)) in testdl:
    preds, bact = model((a, b, e))
    preds = torch.sigmoid(preds)
    activations_zid += zid

    for i, x in enumerate(bact):
        activations[i][mol:mol+bact[0].shape[0], :, :] = bact[i].detach().cpu().numpy()
    mol += bact[0].shape[0]

# ATOM ACTIVATIONS/SUBSTRUCTURES

def remove_duplicates(values, key_lambda):
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet, add it to both list and set.
        cur_key = key_lambda(value)
        if cur_key not in seen:
            output.append(value)
            seen.add(cur_key)
    return output

def get_substructure(atom, radius):
        # Recursive function to get indices of all atoms in a certain radius.
        if radius == 0:
            return set([atom.GetIdx()])
        else:
            cur_set = set([atom.GetIdx()])
            for neighbor_atom in atom.GetNeighbors():
                cur_set.update(get_substructure(neighbor_atom, radius - 1))
            return cur_set
        
def draw(molecule, substructure_idxs, fpix):

    bonds = set()
    for idx in substructure_idxs:
        for idx_1 in substructure_idxs:
            if molecule.GetBondBetweenAtoms(idx, idx_1): bonds.add(molecule.GetBondBetweenAtoms(idx, idx_1).GetIdx())

    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(350,300)
    drawer.drawOptions().fillHighlights=True
    drawer.drawOptions().setHighlightColour((1.0, 0.0, 0.0, 0.2))
    drawer.drawOptions().highlightBondWidthMultiplier=10
    drawer.drawOptions().useBWAtomPalette()
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, highlightAtoms=substructure_idxs, highlightBonds=list(bonds))
    bio = io.BytesIO(drawer.GetDrawingText())
    im = Image.open(bio)
    im = im.save(f"./substructure_activations/fp{fpix}.png")

def plot(activations):
    
    # shape (6, 64, 70, 32)
    # rep (degrees, batch size, atoms with max, hf or fpl?)
    # need to range over mols
    for fpix in range(modelp["fpl"]):
        fpix_list = []
        for mol_ix in range(len(xtest)):
            for rad in range(len(modelp["conv"]["layers"])):
                fp_activations = activations[rad][mol_ix, :, fpix]
                fp_activations = fp_activations[fp_activations != 0]
                fpix_list += [(fp_activation, atom_ix, mol_ix, rad) for atom_ix, fp_activation in enumerate(fp_activations)]
       
        unique_list = remove_duplicates(fpix_list, key_lambda=lambda x: x[0])
        fpix_list = sorted(unique_list, key=lambda x: -x[0])

        try: os.rmdir("./substructure_activations")
        except: pass
        try: os.mkdir("./substructure_activations")
        except: pass

        for fig_ix in range(1):
            # Find the most-activating atoms for this fingerprint index, across all molecules and depths.
            activation, most_active_atom_ix, most_active_mol_ix, ra = fpix_list[fig_ix]
            print(activation, most_active_atom_ix, most_active_mol_ix, ra, activations_zid[most_active_mol_ix])
            ma_smile = smilesdict[activations_zid[most_active_mol_ix]]
            molecule = Chem.MolFromSmiles(ma_smile)
            ma_atom = molecule.GetAtoms()[most_active_atom_ix]
            substructure_idxs = get_substructure(ma_atom, ra)

            draw(molecule, list(substructure_idxs), fpix)

def getTimes():
    jobs = glob.glob(path + 'trainingJobs/*')
    bs_look = 256 
    times = []
    epochs = []
    for job in jobs:
        if bs_look == int(open(job, 'r').readlines()[5].split(' ')[9]):
            i = int(job.split("\\")[1].split(".")[0][5:])
            print(i)
            log = open(path + f"logs/log{i}.txt").readlines()
            epochc = 0
            for line in log:
                if line[:4] == "Time": 
                    times.append(float(line.split(": ")[1]))
                    epochc += 1
            epochs.append(epochc)
    print(statistics.mean(times), statistics.stdev(times), statistics.mean(epochs), statistics.stdev(epochs))

def permutation_importance():
    np.random.seed(0)
    with torch.no_grad():
        a, b, e = buildFeats([x[1] for x in xtest], 6, 70, "perm_test")
        permuted_matrix = [[i, 0] for i in range(a.shape[2])]
        Y = torch.Tensor(ytest).to(device)
        orig_preds, _ = model((a, b, e))
        orig_preds = torch.sigmoid(orig_preds)
        orig_acc = (orig_preds.to(device).round() == Y.to(device)).type(torch.float).sum().item() / orig_preds.shape[0]
        permuted_sum = 0
        for feat_ix in range(a.shape[2]):
            atomfeats = a.clone().detach()
            feature_vector = atomfeats[:, :, feat_ix]
            idx = torch.randperm(feature_vector.nelement())
            feature_vector = feature_vector.view(-1)[idx].view(feature_vector.size())
            atomfeats[:, :, feat_ix] = feature_vector
            permuted_preds, _ = model((atomfeats, b, e))
            permuted_preds = torch.sigmoid(permuted_preds)
            permuted_acc = (permuted_preds.to(device).round() == Y.to(device)).type(torch.float).sum().item() / permuted_preds.shape[0]
            permuted_matrix[feat_ix] = [feat_ix, orig_acc - permuted_acc]
            permuted_sum += orig_acc - permuted_acc
    
    permuted_matrix = [[i[0], i[1]] for i in permuted_matrix]
    permuted_matrix = sorted(permuted_matrix, key=lambda x: -x[1])
    print(permuted_matrix)
    # 0-43 : atoms, 44-49: degree, 50-55: hydrogens, 56-61: implicit valence, 62: aromatic


# SHAP
# batch = next(iter(testdl))
# a, b, e, y = batch
# # f = lambda x: model( torch.Variable( torch.from_numpy(x) ) ).detach().numpy()

# explainer = shap.DeepExplainer(model, (a[:64], b[:64], e[:64]))
# g = torch.Tensor([a[64:68], b[64:68], e[64:68]])
# # shap_values = explainer.shap_values((a[64:68], b[64:68], e[64:68]))

if substr:
    plot(activations=activations)
if tm:
    getTimes()
if permutation:
    permutation_importance()