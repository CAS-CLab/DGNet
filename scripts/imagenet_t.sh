echo "Dynamic dual gating model $1 for ImageNet."

time python main.py -d $2 -dset imagenet -a $1 -lr 0.05 \
     --weight-decay 1e-4 --epochs 100 --checkpoint $5 --gpu-id $4 \
     --den-target $3 --pretrained pytorch --lr-mode cosine
