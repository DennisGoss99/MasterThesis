#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --output=job_output_%j.txt
#SBATCH --mail-type=ALL
#SBATCH -t 24:00:00
#SBATCH -G A100:1

source /home/nwmdgthk/pro/MasterThesis/venv/bin/activate

srun python spiralModel/spiralModelv5.0.1.0.py --path "/scratch/usr/nwmdgthk/allData/Data" -d "AllData_x512" -r 10 -i 5 -o "train/spiralModel" -v "50" -tv "50"