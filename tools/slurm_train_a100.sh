#!/bin/bash
#SBATCH --job-name=ho_recon          # job name
#SBATCH --ntasks=4                   # number of MP tasks
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=6            # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=19:59:59              # maximum execution time (HH:MM:SS)
#SBATCH --output=../outputs/output_logs/%j.out # output file name
#SBATCH --error=../outputs/output_logs/%j.err  # error file name
#SBATCH --signal=USR1@20
#SBATCH -A ets@a100
#SBATCH -C a100

export TMPDIR=$JOBSCRATCH
srun python ./train.py --slurm ${@}
