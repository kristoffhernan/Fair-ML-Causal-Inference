#!/bin/bash

################
# SBATCH OPTIONS
################

#SBATCH --job-name=fairk_test_nofya # job name for queue (optional)
#SBATCH --partition=low    # partition (optional, default=low) 
#SBATCH --error=fairk_test_nofya.err     # file for stderr (optional)
#SBATCH --output=fairk_test_nofya.out    # file for stdout (optional)
#SBATCH --time=4-24:00:00    # max runtime of job hours:minutes:seconds
#SBATCH --nodes=1          # use 1 node
#SBATCH --ntasks=1         # use 1 task
#SBATCH --cpus-per-task=1  # use 1 CPU core
#SBATCH --mail-type=[END,FAIL,INVALID_DEPEND] # send mail when dependency never satisfied
#SBATCH --mail-user=khern045@berkeley.edu 

###################
# Command(s) to run
###################
pip install seaborn

module load python

python -u gen_K_test_nofya.py > gen_K_test_nofya.out
