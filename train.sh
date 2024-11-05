#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=DTR       #Set the job name to "JobExample5"
#SBATCH --time=00:35:00  #Set the wall clock limit to 30 mins for 30 reps with 60 epochs and 10 mins for 9 replications
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=8G                  #Request 2560MB (2.5GB) per node
#SBATCH --output=Output.%j      #Send stdout/err to "Example5Out.[jobID]"
#SBATCH --gres=gpu:1                 #Request 2 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue
##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456             #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=nchapagain@tamu.edu    #Send all emails to email_address 

#First Executable Line
source /sw/eb/sw/Anaconda3/2022.10/etc/profile.d/conda.sh

# Load necessary modules, if required
ml purge
module load GCC/12.2.0 OpenMPI/4.1.4 R/4.3.1

# Activate the Python environment
source ~/.bashrc
source activate in-context-learning

# Navigate to the script directory
cd $SCRATCH/Research/SimulationDirectSearch

python DSAppSim.py 

