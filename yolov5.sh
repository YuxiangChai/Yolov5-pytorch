#!/bin/bash
##
#SBATCH --job-name=v_v5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=v_v5.out
#SBATCH --mail-type=END
#SBATCH --mail-user=yc3743@nyu.edu

module purge
export MPLBACKEND=TKAgg
source /scratch/yc3743/myenv/bin/activate
cd /scratch/yc3743/yolov5
python train.py --model yolov5 --cfg models/yolov5s.yaml
