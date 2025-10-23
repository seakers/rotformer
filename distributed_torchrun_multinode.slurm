#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
# SBATCH --job-name=multinode-multigpu
# SBATCH --nodes=4
# SBATCH --ntasks=8
# SBATCH --time=10:00:00
# SBATCH --mem=5120M
# SBATCH --output=rotformer.%j
# SBATCH --gres=gpu:a100:2
# SBATCH --partition=gpu

ml GCCcore/13.2.0 Python/3.11.5

cd $SCRATCH
source ./torch_venv/bin/activate

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1) 
export MASTER_PORT=12345

srun torchrun \
--nnodes 4 \
--nproc_per_node 2 \
--rdzv_id $SLURM_JOB_ID \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
trainer_distributed_torchrun_multinode.py --dataset=WN18RR --batch_size=1024 --lr=0.0002 --label_smoothing --demb=256 --graph_heads=2 --graph_layer=3 --input_dropout=0.3 --hidden_dropout1=0.4 --hidden_dropout2=0.5