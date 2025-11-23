#!/bin/bash

nohup python -u main.py --model basicvae --latent_dim 256 --mode train --learning_rate 0.001 --batch_size 32 --device 1 &
nohup python -u main.py --model basicvae --latent_dim 256 --mode train --learning_rate 0.0001 --batch_size 32 --device 1 &
nohup python -u main.py --model basicvae --latent_dim 256 --mode train --learning_rate 0.00001 --batch_size 32 --device 1 &
nohup python -u main.py --model basicvae --latent_dim 512 --mode train --learning_rate 0.001 --batch_size 32 --device 1 &
nohup python -u main.py --model basicvae --latent_dim 512 --mode train --learning_rate 0.0001 --batch_size 32 --device 1 &
nohup python -u main.py --model basicvae --latent_dim 512 --mode train --learning_rate 0.00001 --batch_size 32 --device 1 &


nohup python -u main.py --model gan --mode train --learning_rate 0.001 --batch_size 32 --device 0 &
nohup python -u main.py --model gan --mode train --learning_rate 0.0001 --batch_size 32 --device 0 &
nohup python -u main.py --model gan --mode train --learning_rate 0.00001 --batch_size 32 --device 0 &

nohup python -u main.py --model unetgan --mode train --learning_rate 0.001 --batch_size 32 --device 0 &
nohup python -u main.py --model unetgan --mode train --learning_rate 0.0001 --batch_size 32 --device 0 &
nohup python -u main.py --model unetgan --mode train --learning_rate 0.00001 --batch_size 32 --device 0 &

nohup python -u main.py --model resnet --mode train --learning_rate 0.001 --batch_size 32 --device 0 &
nohup python -u main.py --model resnet --mode train --learning_rate 0.0001 --batch_size 32 --device 0 &
nohup python -u main.py --model resnet --mode train --learning_rate 0.00001 --batch_size 32 --device 0 &


nohup python -u main.py --model unet --mode train --learning_rate 0.001 --batch_size 32 --device 0 &
nohup python -u main.py --model unet --mode train --learning_rate 0.0001 --batch_size 32 --device 0 &
nohup python -u main.py --model unet --mode train --learning_rate 0.00001 --batch_size 32 --device 0 &

nohup python -u main.py --model unetvae --latent_dim 256 --mode train --learning_rate 0.001 --batch_size 32 --device 1 &
nohup python -u main.py --model unetvae --latent_dim 256 --mode train --learning_rate 0.0001 --batch_size 32 --device 2 &
nohup python -u main.py --model unetvae --latent_dim 256 --mode train --learning_rate 0.00001 --batch_size 32 --device 3 &
nohup python -u main.py --model unetvae --latent_dim 512 --mode train --learning_rate 0.001 --batch_size 32 --device 1 &
nohup python -u main.py --model unetvae --latent_dim 512 --mode train --learning_rate 0.0001 --batch_size 32 --device 2 &
nohup python -u main.py --model unetvae --latent_dim 512 --mode train --learning_rate 0.00001 --batch_size 32 --device 3 &