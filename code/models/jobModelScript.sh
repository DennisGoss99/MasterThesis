#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --output=job_output_%j.txt
#SBATCH --mail-type=ALL
#SBATCH -t 24:00:00
#SBATCH -G A100:1

source /home/nwmdgthk/pro/MasterThesis/venv/bin/activate

srun python spiralModel/spiralModelv5.0.0.1.py --path "/scratch/usr/nwmdgthk/allData/Data" -d "AllData_1080x" -e -r 20 -i 5 -o "spiralModel"

