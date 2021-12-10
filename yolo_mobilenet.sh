#!/bin/bash
##
#SBATCH --job-name=v_mobile
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=17:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=v_mobile.out
#SBATCH --mail-type=END
#SBATCH --mail-user=yc3743@nyu.edu

module purge
source /scratch/yc3743/myenv/bin/activate
cd /scratch/yc3743/yolov5
python train.py --model yolo_mobilenet --cfg models/yolo_mobilenet.yaml
