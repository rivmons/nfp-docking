import pickle
import torch
import shap
from networkP import dockingProtocol, dockingDataset
from torch.utils.data import DataLoader
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

modelp = None
with open(f'./acease/res/model1/modelparams.pkl', 'rb') as f:
    modelp = pickle.load(f)

testf = open('./acease/res/model1/testset.txt', 'r')
testd = [i.strip().split(",") for i in testf.readlines()[1:]]

smilesf = open('./data/smilesDS.smi', 'r')
smilesd = [i.strip().split(" ") for i in smilesf.readlines()]
smilesdict = {
    i[1]: i[0]
    for i in smilesd
}

for i, mol in enumerate(testd):
    testd[i].append(smilesdict[mol[0]])

# [zid, smile], 0/1
cf = -10
xtest = [[i[0], i[2]] for i in testd]
ytest = [int(float(i[1]) < cf) for i in testd]

model = dockingProtocol(modelp).to(device=device)
model.load_state_dict(torch.load(f'./acease/models/model1.pth'), strict=False)
model.eval()

# testds = dockingDataset(train=xtest,
#                         labels=ytest,
#                         name='test')
# testdl = DataLoader(testds, batch_size=512, shuffle=True)

# batch = next(iter(testdl))
# a, b, e, y = batch
# # f = lambda x: model( torch.Variable( torch.from_numpy(x) ) ).detach().numpy()

# explainer = shap.DeepExplainer(model, (a[:64], b[:64], e[:64]))
# g = torch.Tensor([a[64:68], b[64:68], e[64:68]])
# # shap_values = explainer.shap_values((a[64:68], b[64:68], e[64:68]))