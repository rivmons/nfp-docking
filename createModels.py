import os
import argparse
import glob
import pandas as pd
from networkE import dockingProtocol as dp
from util_reg.networkP import dockingProtocol as dp_reg
from features import num_atom_features
from torch import save
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-ts', '--threshold', required=False)
parser.add_argument('-tm', '--time', required=False)
parser.add_argument('-p', '--protein', required=True)
parser.add_argument('-fpl', '--fplength', required=True)
parser.add_argument('-l', '--left', required=False)
parser.add_argument('-d', '--dataset', required=False)
parser.add_argument('-r', '--regression', required=False)

ioargs = parser.parse_args()
# time = ioargs.time if ioargs.time is not None else str(3840)
# cf = [str(ioargs.threshold) if ioargs.threshold is not None else str(-7.9026)]
protein = ioargs.protein
fplength = ioargs.fplength
lt = ioargs.left
proteinn = protein
dataset = ioargs.dataset
reg = bool(int(ioargs.regression)) if ioargs.regression else False

data = None
if dataset == None: 
    data = 'normal'
    if reg:
        print(f'need to provide name of dataset to do regression (-d); provide file name without .txt suffix')
        sys.exit(1)
else: data = str(dataset)
print(data)

# dropout = [0.3, 0.5, 0.7] # 0.3, 0.5, 0.7
# learn_rate = [0.001, 0.0001, 0.003, 0.0003] # 0.001, 0.0001, 0.003, 0.0003
# weight_decay = [0.0001, 0, 0.001] # 0.0001, 0, 0.001
# oss = [25]
# bs = [64, 128, 256] # was [32, 64] as of acease7

dropout = [0.0]                # static, for testing
learn_rate = [0.001, 0.0001, 0.0003]
weight_decay = [0.0001, 0, 0.001]
oss = [25]
bs = [128, 256]
fpl = [32, 64, 128]
ba = np.array([[2, 4], [2, 4, 8], [2, 8]], dtype=object)
# dropout = [0]
# learn_rate = [0.0003]
# weight_decay = [0.001]
# oss = [25]
# bs = [256]

hps = []
if reg:
    models_hps = [
        [0, 0.001, 0.0001, 25, 64, 32],
        [0, 0.001, 0, 25, 128, 32],
        [0, 0.0001, 0.0001, 25, 256, 64],
        [0, 0.001, 0.0001, 25, 128, 32],
        [0, 0.001, 0.0001, 25, 256, 128]
    ]
    for i in range(10):
        hps.append([
            25,
            np.random.choice(bs),
            0,
            np.random.choice(learn_rate),
            np.random.choice(weight_decay),
            np.random.choice(fpl),
            np.random.choice(ba)
        ])
    
else:
    for ossz in oss:
        for batch in bs:
            for do in dropout:
                for lr in learn_rate:
                    for wd in weight_decay:
                        hps.append([ossz,batch,do,lr,wd])

print(f'num hyperparameters: {len(hps)}')                               

try:
    os.mkdir(f'./{proteinn}')
except:
    pass

try:
    os.mkdir(f'./{proteinn}/trainingJobs')
except:
    pass

try:
    os.mkdir(f'./{proteinn}/logs')
except:
    pass

try:
    os.mkdir(f'./{proteinn}/models')
except:
    pass

try:
    os.mkdir(f'./{proteinn}/res')
except:
    pass

with open(f'./{proteinn}/hpResults.csv', 'w+') as f:
    # f.write(f'{mn},{oss},{bs},{lr},{df},{cf},{fplCmd},{aucValid},{aucPRValid},{precisionValid},{recallValid},{f1Valid},{hitsValid},{aucTest},{aucPRTest},{precisionTest},{recallTest},{f1Test},{hitsTest}\n')
    f.write(f'model number,oversampled size,batch size,learning rate,dropout rate,gfe threshold,fingerprint length,validation auc,validation prauc,validation precision,validation recall,validation f1,validation hits,tr,test auc,test prauc,test precision,test recall,test f1,test hits,avg gfe,t1enrichment,t5enrichment,t10enrichment,t50enrichment,t100enrichment\n')
    

for f in os.listdir(f'./{proteinn}/trainingJobs/'):
    os.remove(os.path.join(f'./{proteinn}/trainingJobs', f))
for f in os.listdir(f'./{proteinn}/models/'):
    os.remove(os.path.join(f'./{proteinn}/models', f))
for f in os.listdir(f'./{proteinn}/logs/'):
    os.remove(os.path.join(f'./{proteinn}/logs', f))
for f in os.listdir(f'./{proteinn}/res/'):
    os.remove(os.path.join(f'./{proteinn}/res', f))

for i in range(len(hps)):
    with open(f'./{proteinn}/trainingJobs/train{i + 1}.sh', 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write(f'cd ./{proteinn}/trainingJobs\n')
        f.write('module load python-libs/3.0\n')
        f.write('source ../../tensorflow_gpu/bin/activate\n')
 
        if reg:
            o,batch,do,lr,wd,fpl,ba = hps[i]
            f.write('python '+'../../reg_train.py'+' '+'-dropout'+' '+str(do)+' '+'-learn_rate'+' '+str(lr)+' '+'-os'+' '+str(o)+' '+'-bs'+' '+str(batch)+' '+'-data '+data+' '+'-fplen '+str(fpl)+' '+'-wd '+str(wd)+' '+'-mnum '+str(i+1)+' '+'-ba '+','.join(list(map(str, ba)))+'\n')
        else:
            o,batch,do,lr,wd = hps[i] 
            f.write('python '+'../../train_actives.py'+' '+'-dropout'+' '+str(do)+' '+'-learn_rate'+' '+str(lr)+' '+'-os'+' '+str(o)+' '+'-bs'+' '+str(batch)+' '+'-protein '+protein+' '+'-fplen '+fplength+' '+'-wd '+str(wd)+' '+'-mnum '+str(i+1)+' '+'-dataset '+data+'\n')


# need to update when updating model params
if reg:
    for i, m in enumerate(hps):
        fpl = int(m[-2]) 
        ba = m[-1]
        print(ba,  [fpl] + list(map(lambda x: int(fpl / x), ba)) + [1])
        hiddenfeats = [fpl] * 4  # conv layers, of same size as fingeprint (so can map activations to features)
        layers = [num_atom_features()] + hiddenfeats
        modelParams = {
            "fpl": fpl,
            "activation": 'regression',
            "conv": {
                "layers": layers
            },
            "ann": {
                "layers": layers,
                "ba": [fpl] + list(map(lambda x: int(fpl / x), ba)) + [1],
                "dropout": 0.1 #arbitrary
            }
        }
        model = dp_reg(modelParams)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'model trainable params: {pytorch_total_params}')
        save(model.state_dict(), f'./{proteinn}/basisModel{i+1}.pth')
else:
    hiddenfeats = [64] * 4 # [32] * 4
    layers = [num_atom_features()] + hiddenfeats 
    fpl = int(fplength) 
    modelParams = {
        "fpl": fpl,
        "batchsize": 0,
        "conv": {
            "layers": layers,
            "activations": False
        },
        "ann": {
            "layers": layers,
            "ba": [fpl, fpl // 4, fpl // 8, 1],  # [fpl, fpl // 4, fpl // 16, 1]
            "dropout": 0.0 # arbitrary
        }
    }
    model = dp(modelParams)
    # print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model trainable params: {pytorch_total_params}')
    save(model.state_dict(), f'./{proteinn}/basisModel.pth')
