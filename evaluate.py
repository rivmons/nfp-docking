import pickle
import torch
from networkE import dockingProtocol, dockingDataset
from util_reg.util import dockingDataset as dd_reg
from util_reg.networkP import dockingProtocol as dp_reg
from util_reg.networkP import EnsembleReg as ep_reg
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
from util_reg.util import buildFeats as bf_reg
import sys

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('-time', '--t', default=False, required=False)
parser.add_argument('-substructure', '--s', default=False, required=False)
parser.add_argument('-permutation', '--p', default=False, required=False)
parser.add_argument('-prediction_file', '--pf', default=False, required=False, help="file path of format (id, smiles) for which to predict molecular property")
parser.add_argument('-model', '--m', default=None, required=False, help="path of model checkpoint for any functionality")

dude = False

args = parser.parse_args()
tm = bool(int(args.t))
substr = bool(int(args.s))
permutation = bool(int(args.p))
predictions = args.pf
model = args.m

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

if predictions:
    print(f'generating predictions for file {predictions}')
    if model == None: 
        print(f'ERROR: need to specify model checkpoint to use for predictions! exiting...')
        sys.exit(1)
    md = None
    scaler = None
    if "ensemble" in model:
        modeld = torch.load(model, map_location=device)
        scaler = modeld["scaler"]
        md = ep_reg(modeld["params"]["num_models"], *(modeld["params"]["models"])).to(device=device)
        md.load_state_dict(modeld['model_state_dict'], strict=False)
    else:
        modeld = torch.load(model, map_location=device)
        scaler = modeld["scaler"]
        md = dp_reg(modeld["params"]).to(device=device)
        md.load_state_dict(modeld['model_state_dict'], strict=False)
    md.eval()
    
    data = [x.strip().split(',') for x in open(predictions, 'r').readlines()]
    if 'smi' in data[0][1]:
        data = data[1:]
    
    testds = dd_reg(train=data,
                    labels=[0] * len(data),
                    name='g_pred')
    testdl = DataLoader(testds, batch_size=512, shuffle=False)
    print('done building tensors')
    preds_s = None
    for (a, b, e, (y, zid)) in testdl:
        preds = md((a, b, e))
        preds = scaler.inverse_transform(preds.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
        if preds_s == None:
            preds_s = preds
        else: preds_s = torch.concatenate(preds, preds_s)
    
    with open(f'predictions_{model.split("/")[0]}.csv', 'w+') as f:
        f.write(f"drug,smile,prediction\n")
        for i in range(len(data)):
            f.write(f'{",".join(data[i])},{preds_s[i]}\n')
    print(f'saved to predictions_{model.split("/")[0]}.csv and exiting')
    sys.exit(0)
    
path = './res_targets/acease/'
mn = 10

modelp = None
with open(f'{path}res/model{mn}/modelparams.pkl', 'rb') as f:
    modelp = pickle.load(f)

testf = open(f'{path}res/model{mn}/testset.txt', 'r')
testd = [i.strip().split(",") for i in testf.readlines()[1:]]

smilesdict = None
xtest, ytest = [], []
actives_d, decoys_d = {}, {}
xhits, xnonhits = [], []
cf = None
zinc_gfe_d = {}    
if not dude:
    smilesf = open('./data/smilesDS.smi', 'r')
    smilesd = [i.strip().split(" ") for i in smilesf.readlines()]
    smilesdict = {
        i[1]: i[0]
        for i in smilesd
    }
    zinc_gfe_d = {
        i[0]: i[1]
        for i in testd
    }
    for i, mol in enumerate(testd):
        testd[i].append(smilesdict[mol[0]])
    cf = -10

    xtest = [[i[0], i[-1]] for i in testd]
    ytest = [int(float(i[1]) < cf) for i in testd]
    xhits = [i[0] for i in testd if float(i[1]) < cf]
    xnonhits = [i[0] for i in testd if float(i[1]) >= cf]
    ###
    # xtest = [['1', 'COC(CC(C)C)c1ccc2cc(-c3ccc(CC(Cc4ccc(C(F)(F)P(=O)(O)O)cc4)(c4ccccc4)n4nnc5ccccc54)cc3)cc(P(=O)(O)O)c2n1']]
    # # ['0', 'COC(=O)c1ccc(C(C/C=C/c2ccccc2)(Cc2ccc(C(F)(F)P(=O)(O)O)cc2)n2nnc3ccccc32)cc1']
    # ytest = [0]
    # # zinc_gfe_d['0'] = 0
    # zinc_gfe_d['1'] = 0
    # # smilesdict['0'] = 'COC(=O)c1ccc(C(C/C=C/c2ccccc2)(Cc2ccc(C(F)(F)P(=O)(O)O)cc2)n2nnc3ccccc32)cc1'
    # smilesdict['1'] = 'COC(CC(C)C)c1ccc2cc(-c3ccc(CC(Cc4ccc(C(F)(F)P(=O)(O)O)cc4)(c4ccccc4)n4nnc5ccccc54)cc3)cc(P(=O)(O)O)c2n1'
    ###
else:
    protein_path = './dude/' + path.split('/')[-2]

    smilesf = open('./data/smilesDS.smi', 'r')
    zincd = [i.strip().split(" ") for i in smilesf.readlines()]
    zincdict = {
        i[1]: i[0]
        for i in zincd
    }

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
        
    smilesdict = {}
    smilesdict.update(actives_d)
    smilesdict.update(decoys_d)
    # testd = id, gfe, label, smile
    for i, mol in enumerate(testd):
        if mol[0][:4] == "ZINC": testd[i].append(zincdict[mol[0]])
        else: testd[i].append(smilesdict[mol[0]])
    xtest = [[i[0], i[3]] for i in testd]
    ytest = [int(i[0] in actives_d) for i in testd]    
    
# [zid, smile], 0/1
# with open(f'./data/ptpase_paper_ligands.csv', 'r') as f:
#     xtest = [x.strip().split(',')[:2] for x in f.readlines()[:]]
#     ytest = [1] * len(xtest)
#     print(xtest, ytest)
    
# smilesdict = {
#         i[0]: i[1]
#         for i in xtest
#     }
# zinc_gfe_d = {
#         i[0]: "active"
#         for i in xtest
#     }
# print(sum(ytest), len(ytest))
# print(smilesdict, zinc_gfe_d)

modelp["conv"]["activations"] = True
model = dockingProtocol(modelp).to(device=device)
model.load_state_dict(torch.load(f'{path}models/model{mn}.pth', map_location=device), strict=False)
model.eval()

testds = dockingDataset(train=xtest,
                        labels=ytest,
                        name='test')
testdl = DataLoader(testds, batch_size=512, shuffle=False)
print('done building tensors')

activations = np.empty((len(modelp["conv"]["layers"]), len(xtest), 200, modelp["fpl"]))
activations_zid = []
mol = 0
for (a, b, e, (y, zid)) in testdl:
    preds, bact = model((a, b, e))
    preds = torch.sigmoid(preds)
    activations_zid += zid

    for i, x in enumerate(bact):
        activations[i][mol:mol+bact[0].shape[0], :, :] = bact[i].detach().cpu().numpy()
    mol += bact[0].shape[0]

if dude:
    # ENRICHMENT FACTOR
    testoutputf = [x.strip().split(',') for x in open(f'{path}res/model{mn}/test_output.txt').readlines()[1:]]
    testlen = len(testoutputf)
    activestest = sum([int(i[2]) for i in testd])
    for percentage in range(1, 21):
        perclen = round(testlen * (percentage / 100))
        actives = 0
        for mol_n in testoutputf[:perclen]:
            if mol_n[0] in actives_d: actives += 1
        print((actives) / ((activestest) * (perclen / testlen)), end=',')
    print()

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
        
def draw(molecule, substructure_idxs, fpix, figix):

    bonds = set()
    for idx in substructure_idxs:
        for idx_1 in substructure_idxs:
            if molecule.GetBondBetweenAtoms(idx, idx_1): bonds.add(molecule.GetBondBetweenAtoms(idx, idx_1).GetIdx())

    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(700,600)
    drawer.drawOptions().fillHighlights=True
    drawer.drawOptions().setHighlightColour((1.0, 0.0, 0.0, 0.2))
    drawer.drawOptions().highlightBondWidthMultiplier=10
    # drawer.drawOptions().useBWAtomPalette()
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, highlightAtoms=substructure_idxs, highlightBonds=list(bonds))
    bio = io.BytesIO(drawer.GetDrawingText())
    im = Image.open(bio)
    im.save(f"./substructure_activations/fp{fpix}_{figix+1}.png")

def plot(activations):
    
    # shape (6, 64, 70, 32)
    # rep (degrees, batch size, atoms with max, hf or fpl?)
    # need to range over mols
    try: os.rmdir("./substructure_activations")
    except: pass
    try: os.mkdir("./substructure_activations")
    except: pass

    of = open('./substructure_activations/activations.txt', 'w')
    of.write('fingerprint_index,rank,fingeprint_index_activation,most_active_atom_ix,most_active_mol_ix,radius,num_atoms,zinc_id,gfe\n')

    all_i = []
    for fpix in range(modelp["fpl"]):
        fpix_list = []
        for mol_ix in range(len(xtest)):
            # if activations_zid[mol_ix] not in xhits: continue
            for rad in range(len(modelp["conv"]["layers"])):
                fp_activations = activations[rad][mol_ix, :, fpix]
                fp_activations = fp_activations[fp_activations != 0]
                fpix_list += [(fp_activation, atom_ix, mol_ix, rad) for atom_ix, fp_activation in enumerate(fp_activations)]
       
        unique_list = remove_duplicates(fpix_list, key_lambda=lambda x: x[0])
        fpix_list = sorted(unique_list, key=lambda x: -x[0])
        
        for fig_ix in range(2):
            # Find the most-activating atoms for this fingerprint index, across all molecules and depths.
            activation, most_active_atom_ix, most_active_mol_ix, ra = fpix_list[fig_ix]
            print(activation, fig_ix, most_active_atom_ix, most_active_mol_ix, ra, activations_zid[most_active_mol_ix])
            ma_smile = smilesdict[activations_zid[most_active_mol_ix]]
            molecule = Chem.MolFromSmiles(ma_smile)
            ma_atom = molecule.GetAtoms()[most_active_atom_ix]
            substructure_idxs = get_substructure(ma_atom, ra)
            of.write(f'{fpix},{fig_ix+1},{activation},{most_active_atom_ix},{most_active_mol_ix},{ra},{len(substructure_idxs)},{activations_zid[most_active_mol_ix]},{zinc_gfe_d[activations_zid[most_active_mol_ix]]}\n')

            i = draw(molecule, list(substructure_idxs), fpix, fig_ix)

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
        for i in range(10):
            for feat_ix in range(a.shape[2]):
                atomfeats = a.clone().detach()
                feature_vector = atomfeats[:, :, feat_ix]
                idx = torch.randperm(feature_vector.nelement())
                feature_vector = feature_vector.view(-1)[idx].view(feature_vector.size())
                atomfeats[:, :, feat_ix] = torch.zeros(feature_vector.shape[0], feature_vector.shape[1])
                permuted_preds, _ = model((atomfeats, b, e))
                permuted_preds_s = torch.sigmoid(permuted_preds)
                permuted_acc = (permuted_preds_s.to(device).round() == Y.to(device)).type(torch.float).sum().item() / permuted_preds_s.shape[0]
                print(permuted_acc)
                permuted_matrix[feat_ix].append([feat_ix, orig_acc - permuted_acc])
                permuted_sum += orig_acc - permuted_acc
    
    print(permuted_matrix)
    permuted_matrix = [[i[0], i[1]] for i in permuted_matrix]
    print(",".join([str(i[1]) for i in permuted_matrix]))
    # permuted_matrix = sorted(permuted_matrix, key=lambda x: -x[1])
    # print(permuted_matrix)
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