#!/bin/bash

ATK=$1

for i in "resnet20_cifar100" "resnet110_cifar100" "densenet40_k12_bc_cifar100" "densenet40_k24_bc_cifar100" "sepreresnet20_cifar100" "sepreresnet56_cifar100" 
do
  python3 main.py --input_dir ./data --output_dir ./results --attack $ATK --model $i
  python3 main.py --eval --input_dir ./data --output_dir ./results --attack $ATK --model $i
done
