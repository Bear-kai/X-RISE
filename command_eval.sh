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
action_dim=20               # joint action (dp3 style): 14,  tcp action (rise style): 20
seed=0                      # keep the same with training seed
num_action=8                # the number of execution steps

data_style='rise'           # 'rise' or 'dp3'; keep the same with training data style
extra_info='_'              # keep the same with training data style

ckpt_path=./checkpoints/${task_name}-${train_env_setting}-${expert_data_num}_${seed}${extra_info}/policy_last.ckpt

PYTHONWARNINGS=ignore::UserWarning,ignore::FutureWarning \
python deploy_policy.py --task_name ${task_name} \
                        --train_env_setting ${train_env_setting} \
                        --eval_env_setting ${eval_env_setting} \
                        --eval_env_seed ${eval_env_seed} \
                        --expert_data_num ${expert_data_num} \
                        --horizon ${horizon} \
                        --num_action ${num_action} \
                        --action_dim ${action_dim} \
                        --ckpt ${ckpt_path} \
                        --obs_feature_dim 512 \
                        --nheads 8 \
                        --dim_feedforward 2048 \
                        --extra_info ${extra_info} \
                        --data_style ${data_style}