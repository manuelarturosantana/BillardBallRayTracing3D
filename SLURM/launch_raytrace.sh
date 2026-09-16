#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=normal

#SBATCH --job-name=RayTrace
#SBATCH --output=%x_%j.out      # e.g. nacelle_123456.out

#SBATCH --time=10-00:00:00     # run time limit (DD-HH:MM:SS)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # This for multiple workers for parallel distributed jobs
#SBATCH --cpus-per-task=64 # This for multi threading. How many CPUS to use
##SBATCH --mem-per-cpu=10G # It may be better to do this differently.
#SBATCH --hint=nomultithread

date 
module load mkl2023.2.0

echo "Running on $(hostname)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#srun --cpu-bind=cores ../build/bounce_map_example /scratch/msantana/PatchFoldersRefined/deleteme/ patch_ 1 134 /scratch/msantana/RayTracing/Plane_by_patchv2.vtk
srun --cpu-bind=cores ../build/bounce_map_example /scratch/msantana/PatchFoldersRefined/PlaneWithNacelleRefinedReordered/ patch_ 1 4212 /scratch/msantana/RayTracing/Plane_by_patchv2.vtk
date




