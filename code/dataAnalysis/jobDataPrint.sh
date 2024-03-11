#!/bin/bash
#SBATCH --partition=large96:shared
#SBATCH --nodes=1
#SBATCH --output=job_output_%j.txt
#SBATCH --mail-type=ALL

source /home/nwmdgthk/pro/MasterThesis/venv/bin/activate

srun python -u plotDataSets_lab.py

