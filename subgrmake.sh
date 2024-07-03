#!/bin/bash -l
#SBATCH --job-name=mkSBgraph
#SBATCH --nodes=1                      
#SBATCH --ntasks=4    
#SBATCH --gres=gpu:0
#SBATCH --time=1-0:00:00
#SBATCH --mem=100G
#SBATCH --partition=week

#SBATCH --output=sbmake.out
#SBATCH --error=sbmake.err

export PYTHONUNBUFFERED=TRUE
module load python-libs

python subgraphs.py