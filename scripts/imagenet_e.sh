echo "Dynamic dual gating model $1 for ImageNet."

time python main.py -d $2 -dset imagenet -a $1 \
     --checkpoint $4 --gpu-id $3 --pretrained $5 -e