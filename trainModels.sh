#!/bin/bash -l
#SBATCH --partition=GPU
#SBATCH --gpus=
#SBATCH --array=
#SBATCH --output=
#SBATCH --time=
##SBATCH --nodes=
##SBATCH --ntasks-per-node=
#SBATCH --mem=
#SBATCH --job-name= 
#SBATCH --mail-user= 
#SBATCH --mail-type=     

export PYTHONUNBUFFERED=TRUE
bash <script> > <log file> 