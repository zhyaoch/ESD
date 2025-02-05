# Offline Goal-Conditioned Reinforcement Learning with Elastic-Subgoal Diffused Policy Learning

<!-- ## [Project Page](https://seohong.me/projects/hiql/) -->

## Overview
This is the official implementation of **Elastic-Subgoal Diffused Policy Learning** (**ESD**).

## Installation

```
conda create --name ESD python=3.8
conda activate ESD
pip install -r requirements.txt --no-deps
pip install "jax[cuda11_cudnn82]==0.4.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install CALVIN (optional)
# Download `calvin.gz` (dataset) following the instructions at https://github.com/clvrai/skimo and place it in the `data` directory.
cd calvin
./install.sh
```

The code relies on [wandb](https://wandb.ai/site/) for logging. Please log in your wandb account following this [document](https://docs.wandb.ai/ref/cli/wandb-login/) before running any experiments.


## Examples

```
# ESD on antmaze-medium-diverse
CUDA_VISIBLE_DEVICES=0 python train.py --run_group Diffusion --seed 0 --env_name antmaze-medium-diverse-v2 --pretrain_steps 1000002 --eval_interval 100000 --save_interval 250000 --p_currgoal 0.2 --p_trajgoal 0.5 --p_randomgoal 0.3 --discount 0.99 --temperature 1 --high_temperature 3 --pretrain_expectile 0.7 --geom_sample 1 --use_layer_norm 1 --value_hidden_dim 512 --value_num_layers 3 --batch_size 1024 --use_rep 1 --policy_train_rep 0 --rep_dim 10 --rep_type concat --algo_name ESD --use_waypoints 1 --way_steps 25 --high_p_randomgoal 0.3

# ESD on antsoccer-arena-navigate
CUDA_VISIBLE_DEVICES=0 python train.py --run_group Diffusion --seed 0 --env_name antsoccer-arena-navigate-v0 --pretrain_steps 1000002 --eval_interval 100000 --save_interval 250000 --p_currgoal 0.2 --p_trajgoal 0.5 --p_randomgoal 0.3 --discount 0.99 --temperature 1 --high_temperature 3 --pretrain_expectile 0.7 --geom_sample 1 --use_layer_norm 1 --value_hidden_dim 512 --value_num_layers 3 --batch_size 1024 --use_rep 1 --policy_train_rep 0 --rep_dim 10 --rep_type concat --algo_name ESD --use_waypoints 1 --way_steps 25 --high_p_randomgoal 0.3

# ESD on scene-play
CUDA_VISIBLE_DEVICES=0 python train.py --run_group Diffusion --seed 0 --env_name scene-play-v0 --pretrain_steps 1000002 --eval_interval 100000 --save_interval 250000 --p_currgoal 0.2 --p_trajgoal 0.5 --p_randomgoal 0.3 --discount 0.99 --temperature 1 --high_temperature 3 --pretrain_expectile 0.7 --geom_sample 1 --use_layer_norm 1 --value_hidden_dim 512 --value_num_layers 3 --batch_size 1024 --use_rep 1 --policy_train_rep 0 --rep_dim 10 --rep_type concat --algo_name ESD --use_waypoints 1 --way_steps 25 --high_p_randomgoal 0.3

# ESD on calvin
CUDA_VISIBLE_DEVICES=0 python train.py --run_group Diffusion --seed 0 --env_name calvin --pretrain_steps 1000002 --eval_interval 100000 --save_interval 250000 --p_currgoal 0.2 --p_trajgoal 0.5 --p_randomgoal 0.3 --discount 0.99 --temperature 1 --high_temperature 3 --pretrain_expectile 0.7 --geom_sample 1 --use_layer_norm 1 --value_hidden_dim 512 --value_num_layers 3 --batch_size 1024 --use_rep 1 --policy_train_rep 0 --rep_dim 10 --rep_type concat --algo_name ESD --use_waypoints 1 --way_steps 25 --high_p_randomgoal 0.3
```
## ✉️ Contact
For any questions, please feel free to email zhangyaocheng2023@ia.ac.cn.

## 🏷️ License

MIT
