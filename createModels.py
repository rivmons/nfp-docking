import os
import argparse
import pandas as pd
from networkP import dockingProtocol
from features import num_atom_features
from torch import save

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--protein', required=True)
parser.add_argument('-fpl', '--fplength', required=True)

ioargs = parser.parse_args()
protein = ioargs.protein
fplength = ioargs.fplength

# dropout = [0.3, 0.5, 0.7]
# learn_rate = [0.001, 0.0001, 0.003, 0.0003]
# weight_decay = [0.0001, 0, 0.001]
# oss = [25]
# bs = [64, 128, 256]

dropout = [0.3]                # static, for testing
learn_rate = [0.001]
weight_decay = [0.0001]
oss = [25]
bs = [64]

hps = []
for ossz in oss:
    for batch in bs:
        for do in dropout:
            for lr in learn_rate:
                for wd in weight_decay:
                    hps.append([ossz,batch,do,lr,wd])

print(f'num hyperparameters: {len(hps)}')                               

try:
    os.mkdir('./src')
except:
    pass

try:
    os.mkdir('./src/trainingJobs')
except:
    pass

try:
    os.mkdir('./src/logs')
except:
    pass

with open('./src/hpResults.csv', 'w+') as f:
    f.write(f'model number,oversampled size,batch size,learning rate,dropout rate,gfe threshold,fingerprint length,validation auc,validation prauc,validation precision,validation recall,validation f1,validation hits,test auc,test prauc,test precision,test recall,test f1,test hits\n')
    

for f in os.listdir('./src/trainingJobs/'):
    os.remove(os.path.join('./src/trainingJobs', f))
for f in os.listdir('./src/logs/'):
    os.remove(os.path.join('./src/logs', f))
    
for i in range(len(hps)):
    with open(f'./src/trainingJobs/train{i + 1}.sh', 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('cd ./src/trainingJobs\n')
        f.write('module load python-libs/3.0\n')
        # f.write('source ../../tensorflow_gpu/bin/activate\n')
 
        o,batch,do,lr,wd = hps[i]
        f.write('python '+'../../subgraph_train.py'+' '+'-dropout'+' '+str(do)+' '+'-learn_rate'+' '+str(lr)+' '+'-os'+' '+str(o)+' '+'-bs'+' '+str(batch)+' '+'-protein '+protein+' '+'-fplen '+fplength+' '+'-wd '+str(wd)+' '+'-mnum '+str(i+1)+'\n')


# need to update when updating model params
fpl = int(fplength) 
hiddenfeats = [fpl] * 4
layers = [num_atom_features()] + hiddenfeats 
modelParams = {
    "fpl": fpl,
    "conv": {
        "layers": layers
    },
    "ann": {
        "layers": layers,
        "ba": [fpl, 1],  # if not doing subgraphs, can be more -- [fpl, fpl // 4, fpl // 16, 1]
        "dropout": 0.0 # arbitrary
    }
}
model = dockingProtocol(modelParams)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'model trainable params: {pytorch_total_params}')
save(model.state_dict(), './src/basisModel.pth')
