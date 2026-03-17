import torch
import argparse
import os
import sys
from datetime import datetime
from termcolor import cprint
import yaml
from pathlib import Path

RISE_ROOT = str(Path(__file__).resolve().parent)
RoboTwin_ROOT = str(Path(__file__).resolve().parent.parent.parent)
# sys.path.append(RISE_ROOT)

from policy.policy import RISE
from env_runner.sim_runner import RobotRunner
from utils.deploy_utils import get_logger, get_ffmpeg, get_robot_cam_config, print_config_infos, class_decorator 


class WrapperPolicyRunner:
    def __init__(self, policy, env_runner) -> None:
        self.policy, self.env_runner = policy, env_runner

    def update_obs(self, observation):
        self.env_runner.update_obs(observation)

    def get_action(self, observation=None):
        action = self.env_runner.get_action(self.policy, observation)
        return action
    
def get_policy_model(args, device):
    cprint("Loading policy ...", "magenta")
    policy = RISE(
        horizon = args.horizon,
        num_action = args.num_action,
        input_dim = 3, # 6,  #
        obs_feature_dim = args.obs_feature_dim,
        action_dim = args.action_dim,
        hidden_dim = args.hidden_dim,
        nheads = args.nheads,
        num_encoder_layers = args.num_encoder_layers,
        num_decoder_layers = args.num_decoder_layers,
        dropout = args.dropout,
        num_obs = args.num_obs,
    ).to(device)

    assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
    policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
    cprint("Checkpoint {} loaded.".format(args.ckpt), "magenta")

    if args.data_style == 'dp3':
        policy.have_normalizer = True

    policy.eval()

    return policy


def encode_obs(observation):
    obs = dict()
    obs['pointcloud'] = observation['pointcloud'][:,:3]   # exclude rgb    
    
    return obs


def eval_func(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    # instruction = TASK_ENV.get_instruction()

    if len(model.env_runner.obs) == 0:      # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)
    actions = model.get_action()            # Get Action according to observation chunk

    for action in actions:                  # Execute each step of the action
        if model.policy.have_normalizer:
            TASK_ENV.take_action(action)                    # dp3 data style: joint action
        else:
            TASK_ENV.take_action(action, action_type='ee')  # rise data style: tcp action
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)               # Update Observation, `update_obs` here can be modified


def reset_func(model):              # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.env_runner.reset_obs()


def eval_policy(TASK_ENV, args, model, st_seed, test_num, video_size):
    global logger
    logger.info(f"Start evalution...")
    logger.info(f"Task Name: {args['task_name']}")
    logger.info(f"Policy Name: {args['policy_name']}")

    expert_check = True
    TASK_ENV.suc = 0        # succeeded number
    TASK_ENV.test_num = 0   # total number

    succ_seed_num = 0
    suc_test_seed_list = []

    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    while succ_seed_num < test_num:
        if expert_check:    
            try:
                TASK_ENV.setup_demo(seed=now_seed, **args)  # now_ep_num=now_id, is_test=True
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except Exception as e:
                print("Error: ", e)
                TASK_ENV.close_env()
                now_seed += 1
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed_num += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            continue

        TASK_ENV.setup_demo(seed=now_seed, **args)

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = get_ffmpeg(TASK_ENV, video_size)
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            # print("\033[92mSuccess!\033[0m")
            logger.info(f"eval_id {TASK_ENV.test_num}: success!")
        else:
            # print("\033[91mFail!\033[0m")
            logger.info(f"eval_id {TASK_ENV.test_num}: fail!")

        TASK_ENV.close_env(clear_cache=((succ_seed_num + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1
        print(f"\033[93m{args['task_name']}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}(eval)\033[0m | \033[91m{args['ckpt_setting']}(train)\033[0m\n"
            + f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 2)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc


def evaluate(args):
    
    # set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    policy = get_policy_model(args, device)
    env_runner = RobotRunner(n_obs_steps=args.num_obs, voxel_size=args.voxel_size)
    wrap_model = WrapperPolicyRunner(policy, env_runner)

    # change working directory
    os.chdir(RoboTwin_ROOT)
    sys.path.append("./")
    from envs import CONFIGS_PATH

    # set params
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = args.task_name
    task_config = args.eval_env_setting
    ckpt_setting = args.train_env_setting
    seed = args.eval_env_seed
    policy_name = args.policy_name

    # get simulated env
    TASK_ENV = class_decorator(task_name) 

    # get config
    with open(os.path.join(CONFIGS_PATH, f"{task_config}.yml"), "r", encoding="utf-8") as f:  # demo_clean.yml
        yaml_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    yaml_args = get_robot_cam_config(yaml_args, CONFIGS_PATH)
    yaml_args["policy_name"] = policy_name
    yaml_args["task_name"] = task_name
    yaml_args["render_freq"] = 0
    yaml_args["task_config"] = task_config
    yaml_args["ckpt_setting"] = ckpt_setting
    print_config_infos(yaml_args)

    # saving directory of evaluation videos
    save_dir = Path(f"{RoboTwin_ROOT}/eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}{args.extra_info}")
    save_dir.mkdir(parents=True, exist_ok=True)
    if yaml_args["eval_video_log"]:
        video_size = f"{yaml_args['head_camera_w']}x{yaml_args['head_camera_h']}"
        yaml_args["eval_video_save_dir"] = save_dir

    # get logger
    global logger
    logger = get_logger(os.path.join(str(save_dir), f"_eval.log"))
        
    # set eval seed
    st_seed = 100000 * (1 + seed)
    test_num = 100 # 1 # 

    # evaluation
    st_seed, suc_num = eval_policy(TASK_ENV, yaml_args, wrap_model, st_seed, test_num, video_size)
    # record results
    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp (start): {current_time}\n")
        file.write(f"Timestamp (end): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        # file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(f"\nsuccess rate: {round(suc_num / test_num * 100, 2)}%")
    logger.info(f"Data has been saved to {file_path}")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action = 'store', type = str, help = 'task name', required = True)
    parser.add_argument('--train_env_setting', action = 'store', type = str, help = 'train environment setting', required = False, default='demo_clean')
    parser.add_argument('--eval_env_setting', action = 'store', type = str, help = 'evaluation environment setting', required = False, default='demo_clean')
    parser.add_argument('--eval_env_seed', action = 'store', type = int, help = 'evaluation environment seed', required = False, default = 0)
    parser.add_argument('--policy_name', action = 'store', type = str, help = 'policy name', required = False, default='RISE')
    parser.add_argument('--expert_data_num', action = 'store', type = int, help = 'number of expert data', required = False, default = 50)
    parser.add_argument('--val_ratio', action = 'store', type = float, help = 'validation ratio', required = False, default = 0.02)
    parser.add_argument('--horizon', action = 'store', type = int, help = 'prediction horizon', required = False, default = 16)
    parser.add_argument('--num_pose', action = 'store', type = int, help = 'number of pose', required = False, default = 8)
    parser.add_argument('--action_dim', action = 'store', type = int, help = 'action dimension', required = False, default = 14)
    parser.add_argument('--num_obs', action = 'store', type = int, help = 'number of observation steps', required = False, default = 1)
    parser.add_argument('--extra_info', action = 'store', type = str, help = 'extra info for video_dir name', required = False, default='')
    parser.add_argument('--data_style', action = 'store', type = str, help = 'dp3 or rise', required = False, default='dp3')
    
    parser.add_argument('--ckpt', action = 'store', type = str, help = 'checkpoint path', required = True)
    parser.add_argument('--num_action', action = 'store', type = int, help = 'number of action steps', required = False, default = 8)
    parser.add_argument('--num_inference_step', action = 'store', type = int, help = 'number of inference query steps', required = False, default = 16)
    parser.add_argument('--voxel_size', action = 'store', type = float, help = 'voxel size', required = False, default = 0.005)
    parser.add_argument('--obs_feature_dim', action = 'store', type = int, help = 'observation feature dimension', required = False, default = 512)
    parser.add_argument('--hidden_dim', action = 'store', type = int, help = 'hidden dimension', required = False, default = 512)
    parser.add_argument('--nheads', action = 'store', type = int, help = 'number of heads', required = False, default = 8)
    parser.add_argument('--num_encoder_layers', action = 'store', type = int, help = 'number of encoder layers', required = False, default = 4)
    parser.add_argument('--num_decoder_layers', action = 'store', type = int, help = 'number of decoder layers', required = False, default = 1)
    parser.add_argument('--dim_feedforward', action = 'store', type = int, help = 'feedforward dimension', required = False, default = 2048)
    parser.add_argument('--dropout', action = 'store', type = float, help = 'dropout ratio', required = False, default = 0.1)
    parser.add_argument('--max_steps', action = 'store', type = int, help = 'max steps for evaluation', required = False, default = 300)
    parser.add_argument('--seed', action = 'store', type = int, help = 'seed', required = False, default = 0)
    parser.add_argument('--vis', action = 'store_true', help = 'add visualization during evaluation')
    parser.add_argument('--discretize_rotation', action = 'store_true', help = 'whether to discretize rotation process.')
    parser.add_argument('--ensemble_mode', action = 'store', type = str, help = 'temporal ensemble mode', required = False, default = 'new')

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    evaluate(args)