#!/bin/bash
# ======================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40g
#SBATCH --time=6-23:59:00

# ======================================================================
# 2022-02-17 DR -- Run Yade example using a singularity image

# ======================================================================
# Specify the main Python program to run with YADE
YADE_SCRIPT=00_brownian_propag_saveA.py
YADE_TABLE=batch_propag.table

echo "Running on host: `hostname`"
echo "Starting at `date`"
echo "Working in directory: `pwd`"

echo "Using singularity: `which singularity`"
singularity --version

IMAGE=/global/software/yade/containers/yade-2019-09-03.sif
echo "Using singularity image: $IMAGE"

COMMAND="yade-batch $YADE_TABLE $YADE_SCRIPT"
echo "Running command: $COMMAND"

# ======================================================================
singularity exec $IMAGE $COMMAND

echo "Finished at `date`"

