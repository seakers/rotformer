#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=firstRun          #Set the job name to "JobExample4"
#SBATCH --time=10:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=5120M                  #Request 2560MB (2.5GB) per task
#SBATCH --output=rotformer.%j        #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:2                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

# modules needed for running DL jobs. Module restore will also work
#module restore dl
ml GCCcore/13.2.0 Python/3.11.5

# Python venv
source $SCRATCH/torch_venv/bin/activate

# scripts or executables
cd $SCRATCH
python trainer_distributed.py --dataset=WN18RR --batch_size=1024 --lr=0.0002 --label_smoothing --demb=256 --graph_heads=2 --graph_layer=3 --input_dropout=0.3 --hidden_dropout1=0.4 --hidden_dropout2=0.5