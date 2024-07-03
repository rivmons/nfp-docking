# Neural Fingerprint-based Molecular Docking

A repository enabling a deep-learning approach to molecular docking. Utilizes differentiable, trainable, neural fingerprints to filter dynamic-sized databases of molecules based on binding favorability to proteins/binding sites. Uses activations to isolate most significant portions of graph.

### steps to run:

##### make sure you have all the necessary python libraries installed with `pip install -r requirements.txt`

1) **use `python createModels.py ...` to populate the src folder with training scripts**
*  `python createModels.py -f/--fpl <fingerprint length> -p/--protein <protein name>`
*   can alter hyperparameters in createModels.py if you'd like; to alter architecture, make
sure you change the architecture in the base model in createModels.py and train.py
</br><br/>

2) **use `sbatch trainModels.sh` if using slurm**
* fill in slurm configuration according to resources available, etc.
* on the last line in `trainModels.sh`, make sure to fill in the path to the log file as well as path to the scripts. Use `$SLURM_ARRAY_TASK_ID` to vary names of log and model based on slurm array
<br/><br/>

3) **results**
* results will be populated in `src/hpResults.csv`. Files with more data are named corresponding to the model number and are withinn `src/trainingJobs/*` with various names
* logs for runs are in `src/logs`

#### Acknowledgements to [Xuhan Liu](https://github.com/XuhanLiu/NGFP) and the work of [Gentile et. al.](https://pubs.acs.org/doi/10.1021/acscentsci.0c00229) for some of the code in this repository