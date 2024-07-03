#!/bin/bash
#SBATCH --output=wrap.out
#SBATCH --error=wrap.err

# wrapper that determines # jobs (can't be dynamically changed once script begins)
num_jobs=$(ls -1 /data/users/vantilme1803/nfp-docking/src/trainingJobs | wc -l)
sbatch --array=1-${num_jobs} trainModels.sh
