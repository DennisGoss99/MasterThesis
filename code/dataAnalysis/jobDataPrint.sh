#!/bin/bash
#SBATCH --partition=large96:shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=job_output_%j.txt
#SBATCH --mail-type=ALL
#SBATCH -t 24:00:00

source /home/nwmdgthk/pro/MasterThesis/venv/bin/activate

srun python plotDataSets.py

