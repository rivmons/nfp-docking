#!/bin/bash -l
#SBATCH --job-name=GCNrun
#SBATCH --nodes=1                      
#SBATCH --ntasks=4    
#SBATCH --gres=gpu:0
#SBATCH --time=1-0:00:00
#SBATCH --mem=100G
#SBATCH --partition=week

export PYTHONUNBUFFERED=TRUE
module load python-libs

joblist=$(ls -1 ./src/trainingJobs)
IFS=$'\n' read -d '' -r -a jobs <<< "$joblist"
job=${jobs[$SLURM_ARRAY_TASK_ID - 1]} # Determine which job this is based on SLURM_ARRAY_TASK_ID
TRAIN_SCRIPT=$"/data/users/vantilme1803/nfp-docking/src/trainingJobs/${job}"

echo "Running job $TRAIN_SCRIPT"
LOG_FILE="/data/users/vantilme1803/nfp-docking/src/logs/log_${SLURM_ARRAY_TASK_ID}.txt"
OUTPUT_FILE="/data/users/vantilme1803/nfp-docking/src/logs/GCN_${SLURM_ARRAY_TASK_ID}.out"
ERROR_FILE="/data/users/vantilme1803/nfp-docking/src/logs/GCN_${SLURM_ARRAY_TASK_ID}.err"

# run job_i with outputs redirected to log file
sbatch --output=$OUTPUT_FILE --error=$ERROR_FILE $TRAIN_SCRIPT > $LOG_FILE 2>&1 
sbatch $TRAIN_SCRIPT $LOG_FILE
echo "RAN JOB $TRAIN_SCRIPT"

