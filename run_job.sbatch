#!/bin/bash
#SBATCH --job-name=clustering_assig2
#SBATCH --time=12:00:00				## total computing time
#SBATCH --nodes=1				## number of nodes
#SBATCH --ntasks-per-node=1			## number of tasks per node
#SBATCH --cpus-per-task=32			## number of CPUs per task
#SBATCH --mem=256GB				## memory per node
#SBATCH --partition=eng-instruction		## queue
#SBATCH --account=25sp-cs598gck-eng		## account
#SBATCH --output=output.log
#SBATCH --error=error.log

#module load python

source /scratch/yusyuan2/env/bin/activate

python ./Clustering.py

deactivate	# end virtual environment
