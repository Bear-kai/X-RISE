# simulated data:
#   single-arm (choose one): beat_block_hammer  # place_shoe   # rotate_qrcode 
#   dual-arm (in sequence):  hanging_mug        # lift_pot
#   dual-arm (in parallel):  place_dual_shoes   # place_bread_skillet  

task_name=beat_block_hammer
train_env_setting=demo_clean
eval_env_setting=demo_clean
eval_env_seed=0
expert_data_num=50

horizon=16                  # keep the same with training horizon
action_dim=20               # keep the same with training action_dim
seed=0                      # keep the same with training seed
num_action=8                # keep the same with training num_action
num_pose=${num_action}

data_style='rise'           # keep the same with training data style
extra_info='_'              # keep the same with training data style
ckpt_path=./checkpoints_X/${task_name}-${train_env_setting}-${expert_data_num}_${seed}${extra_info}/policy_last.ckpt

PYTHONWARNINGS=ignore::UserWarning,ignore::FutureWarning \
python deploy_policy_X.py --task_name ${task_name} \
                        --train_env_setting ${train_env_setting} \
                        --eval_env_setting ${eval_env_setting} \
                        --eval_env_seed ${eval_env_seed} \
                        --expert_data_num ${expert_data_num} \
                        --horizon ${horizon} \
                        --num_action ${num_action} \
                        --num_pose ${num_pose} \
                        --action_dim ${action_dim} \
                        --ckpt ${ckpt_path} \
                        --obs_feature_dim 512 \
                        --nheads 8 \
                        --dim_feedforward 2048 \
                        --data_style ${data_style} \
                        --extra_info ${extra_info}