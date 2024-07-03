#!/bin/bash

#SBATCH --job-name=mkModels
#SBATCH --nodes=1                      
#SBATCH --ntasks=18                    
#SBATCH --gres=gpu:0
#SBATCH --time=4:00:00
#SBATCH --mem=50G
#SBATCH --output=mkMod.out
#SBATCH --error=mkMod.err
#SBATCH --partition=week

module load python-libs
python createModels.py -f 64 -p acease_pruned