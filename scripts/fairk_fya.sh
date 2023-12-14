#!/bin/bash

################
# SBATCH OPTIONS
################

#SBATCH --job-name=fairk_fya # job name for queue (optional)
#SBATCH --partition=low    # partition (optional, default=low) 
#SBATCH --error=fairk_fya.err     # file for stderr (optional)
#SBATCH --output=fairk_fya.out    # file for stdout (optional)
#SBATCH --time=2-24:00:00    # max runtime of job hours:minutes:seconds
#SBATCH --nodes=1          # use 1 node
#SBATCH --ntasks=1         # use 1 task
#SBATCH --cpus-per-task=15  # use 1 CPU core
#SBATCH --mail-type=[END,FAIL,INVALID_DEPEND] # send mail when dependency never satisfied
#SBATCH --mail-user=khern045@berkeley.edu 

###################
# Command(s) to run
###################

# pip install pyro
# pip install scikit-learn

# pip install pyro-ppl
# pip install pandas
# pip install --upgrade pyro-ppl pandas scikit-learn tqdm typing-extensions

module load python/3.10

python -u Fair_ML_fya.py > Fair_ML_fya.out
