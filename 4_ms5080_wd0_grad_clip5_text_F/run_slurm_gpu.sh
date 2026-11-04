#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6      # Cores per node
#SBATCH --partition=gpu          # Partition Name
##
#SBATCH --job-name=gpu
#SBATCH -o test.%N.%j.out         # STDOUT
#SBATCH -e test.%N.%j.err         # STDERR
#
#SBATCH --gres=gpu:rtx4090:1
##

start=$(date +%s)
echo "===== JOB STARTED AT: $(date) ====="

hostname
date

CONFIG_PATH=$1
module purge

python main.py --config "$CONFIG_PATH"

end=$(date +%s)
echo "===== JOB ENDED AT: $(date) ====="
runtime=$((end - start))
hours=$((runtime / 3600))
mins=$(((runtime % 3600) / 60))
secs=$((runtime % 60))
echo "===== TOTAL RUNTIME: ${hours}h ${mins}m ${secs}s ====="
