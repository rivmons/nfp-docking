import os
import argparse
import glob
import pandas as pd
from networkE import dockingProtocol
from features import num_atom_features
from torch import save

parser = argparse.ArgumentParser()
parser.add_argument('-ts', '--threshold', required=False)
parser.add_argument('-tm', '--time', required=False)
parser.add_argument('-p', '--protein', required=True)
parser.add_argument('-fpl', '--fplength', required=True)
parser.add_argument('-l', '--left', required=False)
parser.add_argument('-d', '--dataset', required=False)

ioargs = parser.parse_args()
# time = ioargs.time if ioargs.time is not None else str(3840)
# cf = [str(ioargs.threshold) if ioargs.threshold is not None else str(-7.9026)]
protein = ioargs.protein
fplength = ioargs.fplength
lt = ioargs.left
proteinn = protein
dataset = ioargs.dataset

datasetname = None
if dataset == None: datasetname = 'normal'
else: datasetname = str(dataset)
print(datasetname)

# dropout = [0.3, 0.5, 0.7] # 0.3, 0.5, 0.7
# learn_rate = [0.001, 0.0001, 0.003, 0.0003] # 0.001, 0.0001, 0.003, 0.0003
# weight_decay = [0.0001, 0, 0.001] # 0.0001, 0, 0.001
# oss = [25]
# bs = [64, 128, 256] # was [32, 64] as of acease7

dropout = [0]
learn_rate = [0.001, 0.0003, 0.0001]
weight_decay = [0, 0.001, 0.0001]
oss = [25]
bs = [256, 128]
# dropout = [0]
# learn_rate = [0.0003]
# weight_decay = [0.001]
# oss = [25]
# bs = [256]

hps = []
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
 
        o,batch,do,lr,wd = hps[i]
        f.write('python '+'../../train.py'+' '+'-dropout'+' '+str(do)+' '+'-learn_rate'+' '+str(lr)+' '+'-os'+' '+str(o)+' '+'-bs'+' '+str(batch)+' '+'-protein '+protein+' '+'-fplen '+fplength+' '+'-wd '+str(wd)+' '+'-mnum '+str(i+1)+' '+'-dataset '+datasetname+'\n')


# need to update when updating model params
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
model = dockingProtocol(modelParams)
# print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'model trainable params: {pytorch_total_params}')
save(model.state_dict(), f'./{proteinn}/basisModel.pth')