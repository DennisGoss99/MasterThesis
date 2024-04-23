#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --output=job_output_%j.txt
#SBATCH -t 24:00:00
#SBATCH -G A100:4

source /home/nwmdgthk/pro/MasterThesis/venv/bin/activate

export MASTER_ADDR=localhost
export MASTER_PORT=12355

srun python rowModel/rollModelv2.1.8.1.py --path "/scratch/usr/nwmdgthk/allData/Data" -d "AllData_x512" -r 1 -i 10 -o "train/bigColumnModel" -v "2%" -tv "2%"