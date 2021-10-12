echo "Dynamic dual gating model $1 for cifar-10."

time python main.py -d $2 -dset cifar10 -j 2 -a $1 -b 128 \
     --checkpoint $4 --gpu-id $3 --pretrained $5 -e
