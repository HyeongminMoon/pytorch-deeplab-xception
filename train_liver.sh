CUDA_VISIBLE_DEVICES=0 python train.py --backbone xception --lr 0.1 --workers 4 --epochs 50 --batch-size 16 --gpu-ids 0 --checkname deeplab-xception-liver-bce --eval-interval 1 --loss-type ce --dataset lits_liver

## epoch당 36분정도