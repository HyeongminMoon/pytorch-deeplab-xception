CUDA_VISIBLE_DEVICES=0 python train.py --backbone xception --lr 0.007 --workers 4 --epochs 50 --batch-size 4 --gpu-ids 0 --checkname deeplab-xception --eval-interval 1 --loss-type ce --dataset pascal
