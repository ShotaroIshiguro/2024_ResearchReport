#!/bin/bash

echo "deca resnet18 running..."
mpiexec -n 2 
python scripts/train.py --latent_dim 64 --encoder_type resnet18 \
    --log_dir log_FFHQ/stage1/deca_FFHQ_resnet18 --data_dir ffhq256_deca.lmdb --lr 1e-4 \
    --p2_weight True --image_size 256 --batch_size 8 --max_steps 100000 \
    --num_workers 8 --save_interval 10000 --stage 1

echo "deca resnet50 running..."

mpiexec -n 2
python scripts/train.py --latent_dim 64 --encoder_type resnet50 \
    --log_dir log_FFHQ/stage1/deca_FFHQ_resnet50 --data_dir ffhq256_deca.lmdb --lr 1e-4 \
    --p2_weight True --image_size 256 --batch_size 8 --max_steps 100000 \
    --num_workers 8 --save_interval 10000 --stage 1

echo "emoca resnet18 runnning..."

mpiexec -n 2
python scripts/train.py --latent_dim 64 --encoder_type resnet18 \
    --log_dir log_FFHQ/stage1/emoca_FFHQ_resnet18 --data_dir ffhq256_emoca.lmdb --lr 1e-4 \
    --p2_weight True --image_size 256 --batch_size 8 --max_steps 100000 \
    --num_workers 8 --save_interval 10000 --stage 1

echo "emoca resnet50 runnning..."

mpiexec -n 2
python scripts/train.py --latent_dim 64 --encoder_type resnet50 \
    --log_dir log_FFHQ/stage1/emoca_FFHQ_resnet50 --data_dir ffhq256_emoca.lmdb --lr 1e-4 \
    --p2_weight True --image_size 256 --batch_size 8 --max_steps 100000 \
    --num_workers 8 --save_interval 10000 --stage 1



