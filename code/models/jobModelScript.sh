#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --output=job_output_%j.txt
#SBATCH --mail-type=ALL
#SBATCH -t 24:00:00
#SBATCH -G A100:1

source /home/nwmdgthk/pro/MasterThesis/venv/bin/activate

srun python rowModel/rollModelv2.1.8.0.py --path "/scratch/usr/nwmdgthk/allData/Data" -d "AllData_x512" -r 10 -i 5 -o "train/columnModel" -v "5%" -tv "5%"