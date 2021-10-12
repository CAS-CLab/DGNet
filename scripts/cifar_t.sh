echo "Dynamic dual gating model $1 for cifar-10."

dgnet_cifar10(){
time python main.py -d $2 -dset cifar10 -j 2 -a $1 -b 128 -lr 0.1 \
     --weight-decay 5e-4 --schedule 150 225 --checkpoint $5 \
     --gpu-id $4 --den-target $3 --alpha 2e-2 --pretrained $6
}

checkpoint1="$5_varience1"
checkpoint2="$5_varience2"
checkpoint3="$5_varience3"

dgnet_cifar10 $1 $2 $3 $4 $checkpoint1 $6
dgnet_cifar10 $1 $2 $3 $4 $checkpoint2 $6
dgnet_cifar10 $1 $2 $3 $4 $checkpoint3 $6
