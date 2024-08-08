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
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score, fbeta_score
import matplotlib.pyplot as plt
import dill
import psutil
import sys
from networkP import dockingProtocol
from util import buildFeats
import time
import copy

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('-dropout','--df',required=True)
parser.add_argument('-learn_rate','--lr',required=True)
parser.add_argument('-os','--os',required=True)
parser.add_argument('-protein', '--pro', required=True)
parser.add_argument('-bs', '--batch_size', required=True)
parser.add_argument('-fplen', '--fplength', required=True)
parser.add_argument('-mnum', '--model_number', required=True)
parser.add_argument('-wd', '--weight_decay', required=True)

cmdlArgs = parser.parse_args()
df=float(cmdlArgs.df)
lr=float(cmdlArgs.lr)
oss=int(cmdlArgs.os)
wd=float(cmdlArgs.weight_decay)
bs=int(cmdlArgs.batch_size)
protein = cmdlArgs.pro
fplCmd = int(cmdlArgs.fplength)
mn = cmdlArgs.model_number

print('hyperparameters: ', end='')
print(df,lr,wd, sep=', ')
print(f'interop threads: {torch.get_num_interop_threads()}, intraop threads: {torch.get_num_threads()}')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

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


def labelsToDF(fname):
    arr = []
    with open(fname) as f:
        for line in f.readlines():
            try:
                l = None
                try:
                    l = [float(line.split('\t')[0].split(' ')[2])]
                except:
                    l = [float(line.split('\t')[1])]
                if l[0] >= 10: continue
                zid = line.split('\t')[2].rstrip() if "ZINC" not in line.split('\t')[1].rstrip() else line.split('\t')[1].rstrip()
                l.append(zid)
                arr.append(l)
            except:
                continue
    df = pd.DataFrame(arr)
    df.columns = ['labels', 'zinc_id']
    return df

# 70-15-15 split
allData = labelsToDF(f'../../data/dock_{protein}.txt')
allData.set_index('zinc_id', inplace=True)
trainData, validationData, testData = np.split(allData.sample(frac=1), 
                                        [int(.70*len(allData)), int(.85*len(allData))])

print(f'merged df shapes: {trainData.shape}, {validationData.shape}, {testData.shape}')

trainData = pd.DataFrame(trainData)
validationData = pd.DataFrame(validationData)
testData = pd.DataFrame(testData)
smileData = pd.read_csv('../../data/smilesDS.smi', delimiter=' ')
smileData.columns = ['smile', 'zinc_id']
smileData.set_index('zinc_id', inplace=True)

gfeDist = allData['labels'].to_numpy()
print(f'mean,std (gfe): {np.mean(gfeDist)}, {np.std(gfeDist)}')
tl = 5000 # alter this
pr = tl / len(smileData)
gfeDist = np.sort(gfeDist)
cf = gfeDist[int(0.20 * gfeDist.shape[0])] # gfeDist[int(pr * gfeDist.shape[0])]
print(f'cf = {cf}')

yTrain = trainData.loc[:,'labels']<cf

allPD = []
allLabels = []
yHit = yTrain[yTrain.values==1]
yNHit = yTrain[yTrain.values==0]
hitC = yHit.shape[0]
nhitC = yNHit.shape[0]
print(f'hits in dataset: {hitC}, non-hits in dataset: {nhitC}')

oversampleSize = np.min([nhitC, 50000, hitC*oss*8])
print(f'sample size for oversampled dataset: {oversampleSize}')

trainTuples = []
for i in range(oversampleSize):
    iPos = random.randint(0, hitC-1)
    iNeg = random.randint(0, nhitC-1)
    # 50-50 balanced
    trainTuples.append((yHit.index[iPos], 1))
    trainTuples.append((yNHit.index[iNeg], 0))

random.shuffle(trainTuples)
xValidL = validationData.index.tolist()
yValid = (validationData.loc[:, 'labels']<cf).astype(int).to_numpy().tolist()
xTestL = testData.index.tolist()
yTest = (testData.loc[:, 'labels']<cf).astype(int).to_numpy().tolist()
yHit = []
yNHit = []

trainL = pd.DataFrame(trainTuples)
trainL.columns = ['zinc_id', 'labels']
trainL.set_index('zinc_id', inplace=True)
trainL = pd.merge(trainL, smileData, on='zinc_id')
validL = (validationData.loc[:, 'labels']<cf).astype(int).reset_index()
testL = (testData.loc[:, 'labels']<cf).astype(int).reset_index()
validL.columns = ['zinc_id', 'labels']
testL.columns = ['zinc_id', 'labels']
validL.set_index('zinc_id', inplace=True)
testL.set_index('zinc_id', inplace=True)
validL = pd.merge(validL, smileData, on='zinc_id')
testL = pd.merge(testL, smileData, on='zinc_id')

xTrain = trainL.reset_index()[['zinc_id', 'smile']].values.tolist()
yTrain = [l[0] for l in trainL.reset_index()[['labels']].values.tolist()]
xValid = validL.reset_index()[['zinc_id', 'smile']].values.tolist()
yValid = [l[0] for l in validL.reset_index()[['labels']].values.tolist()]
xTest = testL.reset_index()[['zinc_id', 'smile']].values.tolist()
yTest = [l[0] for l in testL.reset_index()[['labels']].values.tolist()]

hiddenfeats = [32] * 4 # 32
layers = [num_atom_features()] + hiddenfeats 
fpl = fplCmd 
modelParams = {
    "fpl": fpl,
    "conv": {
        "layers": layers
    },
    "ann": {
        "layers": layers,
        "ba": [fpl, fpl // 4, fpl // 16, 1], # fpl, fpl // 4, 1
        "dropout": df
    }
}
print(f'layers: {layers}, through-shape: {list(zip(layers[:-1], layers[1:]))}')

trainds = dockingDataset(train=xTrain, 
                        labels=yTrain,
                        name='train')
traindl = DataLoader(trainds, batch_size=bs, shuffle=True)
testds = dockingDataset(train=xTest,
                        labels=yTest,
                        name='test')
testdl = DataLoader(testds, batch_size=bs, shuffle=True)
validds = dockingDataset(train=xValid,
                         labels=yValid,
                         name='valid')
validdl = DataLoader(validds, batch_size=bs, shuffle=True)

model = dockingProtocol(modelParams).to(device=device)
print(model)
# print("inital grad check")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
totalParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'total trainable params: {totalParams}')
print(sum(yTrain), len(yTrain))
lossFn = nn.BCELoss()
# adam, lr=0.01, weight_decay=0.001, prop=0.2, dropout=0.2
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
model.load_state_dict(torch.load('../basisModel.pth'), strict=False)
lendl = len(trainds)
bestVLoss = 100000000
bestmodel = None
lastEpoch = False
epochs = 200  # 200 initially 
cepoch = 0
earlyStop = EarlyStopper(patience=50, min_delta=0.01)
trainLoss, validLoss = [], []
for epoch in range(1, epochs + 1):
    cepoch = epoch
    print(f'\nEpoch {epoch}\n------------------------------------------------')
    
    stime = time.time()
    model.train()
    runningLoss, corr = 0, 0

    for batch, (a, b, e, (y, zidTr)) in enumerate(traindl):
        at, bo, ed, Y = a.to(device), b.to(device), e.to(device), y.to(device)

        preds = model((at, bo, ed))
        loss = lossFn(preds, Y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        corr += (preds.round() == Y).type(torch.float).sum().item()
        runningLoss += preds.shape[0] * loss.item()
   
        if batch % (np.ceil(lendl / bs / 10)) == 0:
            lossDisplay, currentDisplay = loss.item(), (batch + 1)
            print(f'loss: {lossDisplay:>7f} [{((batch + 1) * len(a)):>5d}/{lendl:>5d}]')

    trainLoss.append(runningLoss/lendl)
    print(f'Time to complete epoch: {time.time() - stime}')
    print(f'\nTraining Epoch {epoch} Results:\nacc: {((100 * (corr/lendl))):>0.1f}%, loss: {runningLoss/lendl:>8f}\n------------------------------------------------')
    
    size = len(validdl.dataset)
    num_batches = len(validdl)
    model.eval()
    valid_loss, correct = 0, 0
    with torch.no_grad():
        for (a, b, e, (y, zidValid)) in validdl:
            preds = model((a, b, e))
            valid_loss += lossFn(preds.to(device), y.to(device)).item()
            correct += (preds.to(device).round() == y.to(device)).type(torch.float).sum().item()
    valid_loss /= num_batches
    correct /= size
    validLoss.append(valid_loss)
    print(f'\nValidation Results:\nacc: {(100*correct):>0.1f}%, loss: {valid_loss:>8f}\n------------------------------------------------')
    
    if valid_loss < bestVLoss:
        bestVLoss = valid_loss
        bestmodel = copy.deepcopy(model.state_dict())
        print(f"best model checkpointed at epoch {epoch}")

    if earlyStop.early_stop(valid_loss):
        print(f'validation loss converged to ~{valid_loss}')
        break

epochR = range(1, cepoch + 1)
plt.plot(epochR, trainLoss, label='Training Loss')
plt.plot(epochR, validLoss, label='Validation Loss')
 
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
 
plt.xticks(np.arange(0, epochs + 1, 2))
 
plt.legend(loc='best')
plt.savefig(f'../res/loss{mn}.png')
plt.show()
plt.close()
with open(f'../res/lossData{mn}.txt', 'w+') as f:
    f.write('train loss, validation loss\n')
    f.write(f'{",".join([str(x) for x in trainLoss])}\n')
    f.write(f'{",".join([str(x) for x in validLoss])}')

try:
    torch.save(bestmodel, f'../models/model{mn}.pth')
    bmodel = dockingProtocol(modelParams).to(device=device)
    bmodel.load_state_dict(torch.load(f'../models/model{mn}.pth'), strict=False)
    bmodel.eval()
    yVal, predVal = [], []
    print(f'final validation using validation set of shape {np.array(yValid).shape}')             
    with torch.no_grad():
        for (a, b, e, (y, zidValidF)) in validdl:
            preds = bmodel((a, b, e))
            predVal += preds.tolist()
            yVal += y.tolist()
    
    thresholds = np.arange(0, 1, 0.001)
    scores = [fbeta_score(yVal, (np.array(predVal) >= t).astype('int'), beta=1.75) for t in thresholds]
    bestIx = np.argmax(scores)
    tr = thresholds[bestIx]
    print(f'threshold={tr:.3f}, f1={scores[bestIx]:.5f}')
    precValid, recValid, thresholdsValid = precision_recall_curve(np.array(yVal), np.array(predVal))
    scores2 = [fbeta_score(yVal, (np.array(predVal) >= t).astype('int'), beta=1.75) for t in thresholdsValid]
    bestIx2 = np.argmax(scores2)
    tr2 = thresholdsValid[bestIx2]
    print(f'pr-based -> threshold={tr2:.3f}, f1={scores2[bestIx2]:.5f}')
    fprValid, tprValid, threshValid = roc_curve(yVal, predVal)
    aucValid = auc(fprValid, tprValid)
    aucPRValid = average_precision_score(yVal, predVal)
    hitsValid = np.sum(yVal)
    precisionValid = precision_score(yVal, [int(x >= tr) for x in predVal], average='binary')
    recallValid = recall_score(yVal, [int(x >= tr) for x in predVal], average='binary')
    f1Valid = f1_score(yVal, [int(x >= tr) for x in predVal], average='binary')
    print(aucValid, aucPRValid, precisionValid, recallValid, f1Valid)
    
    print(f'final testing using testing set of shape {np.array(yTest).shape}') 
    testData.reset_index(inplace=True)  
    yTst, predTst = [], []
    pos, neg = [], []
    bin_o = {}
    gfe_delta = []
    with torch.no_grad():
        for(a, b, e, (y, zidTe)) in testdl:
            pred = bmodel((a, b, e))
            predTst += pred.tolist()
            yTst += y.tolist()
            for i, P in enumerate(pred.tolist()):
                if P >= tr: 
                    pos.append(zidTe[i])
                    gfe_delta.append(testData.loc[testData['zinc_id'] == zidTe[i]].values[0][1])
                else: neg.append(zidTe[i])
                bin_o[zidTe[i]] = P
    precTest, recTest, thresholdsTest = precision_recall_curve(np.array(yTst), np.array(predTst))
    fprTest, tprTest, threshTest = roc_curve(yTst, predTst)
    aucTest = auc(fprTest, tprTest)
    aucPRTest = average_precision_score(yTst, predTst)
    hitsTest = np.sum(yTst)
    precisionTest = precision_score(yTst, [int(x >= tr) for x in predTst], average='binary')
    recallTest = recall_score(yTst, [int(x >= tr) for x in predTst], average='binary')
    f1Test = f1_score(yTst, [int(x >= tr) for x in predTst], average='binary')
    print(aucTest, aucPRTest, precisionTest, recallTest, f1Test)
    print(confusion_matrix(yTst, [int(x >= tr) for x in predTst]))
    tn, fp, fn, tp = confusion_matrix(yTst, [int(x >= tr) for x in predTst]).ravel()
    avgGfe = np.mean(gfe_delta)
    with open(f'../res/trData{mn}.txt', 'w+') as f:
        f.write("tr,prec,rec,f1,fb(b=1.75)\n")
        for triter in thresholds.tolist():
            try:
                p = precision_score(yTst, [int(x >= triter) for x in predTst], average='binary')
                r = recall_score(yTst, [int(x >= triter) for x in predTst], average='binary')
                fm = f1_score(yTst, [int(x >= triter) for x in predTst], average='binary')
                fb = fbeta_score(yTst, [int(x >= triter) for x in predTst], average='binary', beta=2)
                f.write(f'{triter},{p},{r},{fm}\n')
            except:
                continue
     
    with open(f'../res/miscData{mn}.txt', 'w+') as f: 
        f.write('AUC data (thresholds, fpr, tpr)\n') 
        f.write(f"{','.join([str(x) for x in threshTest.tolist()])}\n")
        f.write(f"{','.join([str(x) for x in fprTest.tolist()])}\n")
        f.write(f"{','.join([str(x) for x in tprTest.tolist()])}\n") 
        f.write('prAUC data (rec, prec)\n') 
        f.write(f"{','.join([str(x) for x in recTest.tolist()])}\n") 
        f.write(f"{','.join([str(x) for x in precTest.tolist()])}\n") 
        f.write(f"confusion matrix (tn, fp, fn, tp)\n") 
        f.write(f'{tn},{fp},{fn},{tp}\n')
        f.write("gfe of virtual hits\n")
        f.write(f"{','.join([str(x) for x in gfe_delta])}")
     
    enrichDist = testData.loc[testData['labels'] < cf]
    enrichment = enrichDist.loc[enrichDist['zinc_id'].isin(pos)]
    cfLow = testData['labels'].min()
    probs = []
    for i in range(int(cfLow * 100), int((cf + 0.01) * 100)):
        s1 = enrichDist.loc[enrichDist['labels'] == np.round(i/100, 2)]
        s2 = enrichment.loc[enrichment['labels'] == np.round(i/100, 2)]
        if s1.empty and s2.empty: continue
        probs.append((np.round(i / 100, 2), s1.shape[0], s2.shape[0]))
    with open(f'../res/enrichmentProbs{mn}.txt', 'w+') as f: 
        f.write('prob, total mols w gfe value, total mols predicted as pos w gfe value\n')
        f.write(f"{','.join([str(x[0]) for x in probs])}\n")
        f.write(f"{','.join([str(x[1]) for x in probs])}\n")
        f.write(f"{','.join([str(x[2]) for x in probs])}\n") 
    
    bin_sorted = {k: v for k, v in sorted(bin_o.items(), key=lambda item: item[1], reverse=True)}
    with open(f'../res/test_output_{mn}.txt', 'w+') as f:
        f.write("zid,output\n")
        for vh in bin_sorted:
            f.write(f'{vh},{bin_sorted[vh]}\n')
    
    plt.plot([x[0] for x in probs], [x[2]/x[1] for x in probs], 'b--', label='pdf of enrichment')
    plt.ylim([0, 1.2])
    plt.savefig(f'../res/enrichment{mn}.png')
    plt.show()
    plt.close()

    plt.plot(fprTest, tprTest, label=f'ROC curve (area = {aucTest:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'../res/rocCurve_{mn}.png')
    plt.show()
    plt.close()
    
    plt.plot(recTest, precTest, marker='o', color='darkorange', lw=2, label='PR Curve (AUC = %0.2f)' % aucPRTest)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'../res/prCurve_{mn}.png')
    plt.show()
 
    with open('../hpResults.csv','a') as f:
        f.write(f'{mn},{oss},{bs},{lr},{df},{cf},{fplCmd},{aucValid},{aucPRValid},{precisionValid},{recallValid},{f1Valid},{hitsValid},{tr},{aucTest},{aucPRTest},{precisionTest},{recallTest},{f1Test},{hitsTest},{avgGfe}\n')
except:
    pass