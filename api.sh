#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:30:00
#SBATCH --job-name=jupyter-notebook
#SBATCH --output=jupyter_%j.out
#SBATCH --error=jupyter_%j.err

# get tunneling info
XDG_RUNTIME_DIR="/home/vikash06/src/"
node=$(hostname -s)
user=$(whoami)
cluster="schooner"
port=8005

# print tunneling instructions jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}.oscer.ou.edu

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
#module load anaconda3/2020.11
# module load Anaconda3
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   
# Run Jupyter
# /home/vikash06/miniconda3/envs/api/bin/python train.py

/home/vikash06/miniconda3/envs/api/bin/python api_model.py
