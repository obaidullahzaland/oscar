#!/bin/bash
#SBATCH -A Berzelius-2024-55
#SBATCH --gpus 1
#SBATCH -t 3-00:00:00

conda activate oneshot

python train.py --round 1 --backbone "resnet101"
python train.py --round 2 --backbone "resnet101"


python train.py --round 1 --dataset "nico_u" --backbone "resnet101"
python train.py --round 2 --dataset "nico_u" --backbone "resnet101"


python train.py --round 1 
python train.py --round 2 



python train.py --round 1 --dataset "nico_u"
python train.py --round 2 --dataset "nico_u"
# python train.py --round 3 --dataset "nico_u"
# python train.py --round 4 --dataset "nico_u"
# python train.py --round 5 --dataset "nico_u"





