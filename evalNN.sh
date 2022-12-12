#!/bin/bash
# Author: Stephen Toner
# Batch Job Settings:

#SBATCH --job-name=lstm_model_eval
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=25GB
#SBATCH --time=05:00:00
#SBATCH --account=stats_dept2
#SBATCH --partition=standard

# Run your program
#(">" redirects the output of your program, 
# in this case to "output.txt")

# n_procs=5

python3 eval_lstm.py calibration > calibration.txt