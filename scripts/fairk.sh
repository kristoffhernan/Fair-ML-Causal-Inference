#!/bin/bash


################
# SBATCH OPTIONS
################

#SBATCH --job-name=fairk # job name for queue (optional)
#SBATCH --partition=low    # partition (optional, default=low) 
#SBATCH --error=fairk.err     # file for stderr (optional)
#SBATCH --output=fairk.out    # file for stdout (optional)
#SBATCH --time=2-24:00:00    # max runtime of job hours:minutes:seconds
#SBATCH --nodes=1          # use 1 node
#SBATCH --ntasks=1         # use 1 task
#SBATCH --cpus-per-task=1  # use 1 CPU core
#SBATCH --mail-type=[END,FAIL,INVALID_DEPEND] # send mail when dependency never satisfied
#SBATCH --mail-user=khern045@berkeley.edu 

###################
# Command(s) to run
###################

%pip install pyro
%pip install scikit-learn
%pip install tqdm
%pip install seaborn

module load python

python -u Fair-ML.py > Fair-ML.out
