echo "Dynamic dual gating model $1 for ImageNet."

time python main.py -d $2 -dset imagenet -a $1 -lr 0.05 \
     --weight-decay 4e-5 --epochs 200 --checkpoint $5 \
     --gpu-id $4 --den-target $3 --pretrained pytorch \
     --lr-mode cosine
