#!/bin/bash
#SBATCH --partition=large96:shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=job_output_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dennisgossler98@gmail.com

source ~/pro/MasterThesis/venv/bin/activate

srun python plotDataSets.py

