# simulated data:
#   single-arm (choose one): beat_block_hammer  # place_shoe   # rotate_qrcode 
#   dual-arm (in sequence):  hanging_mug        # lift_pot
#   dual-arm (in parallel):  place_dual_shoes   # place_bread_skillet  
#
# real data: hang_mug_sampled, pour_balls_sampled, arrange_truck_sampled

task_name=beat_block_hammer
train_env_setting=demo_clean
expert_data_num=50

horizon=16 
action_dim=20           # joint action (dp3 style): 14,  tcp action (rise style): 20,  real-world (same as rise style): 10
seed=0

data_style='rise'       # 'rise', 'dp3', 'real'
aug_prob=1.0            # probability of data augmentation, only effective for rise data style
extra_info='_'

CUDA_VISIBLE_DEVICES=0 \

python train.py --task_name ${task_name} \
                --train_env_setting ${train_env_setting} \
                --expert_data_num ${expert_data_num} \
                --val_ratio 0 \
                --horizon ${horizon} \
                --action_dim ${action_dim} \
                --lr 1e-4 \
                --batch_size 128 \
                --num_epochs 1000 \
                --seed ${seed} \
                --data_style ${data_style} \
                --obs_feature_dim 512 \
                --nheads 8 \
                --dim_feedforward 2048 \
                --aug ${aug_prob} \
                --extra_info ${extra_info}