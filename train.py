import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import os 
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"        # Intel MKL
os.environ["OMP_NUM_THREADS"] = "1"        # OpenMP
os.environ["NUMEXPR_NUM_THREADS"] = "1"    # NumExpr

import torch
import argparse
import MinkowskiEngine as ME
from termcolor import cprint

from tqdm import tqdm
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

import yaml
import pathlib
RISE_ROOT = str(pathlib.Path(__file__).resolve().parent)
RoboTwin_ROOT = str(pathlib.Path(__file__).resolve().parent.parent.parent)

from policy.policy import RISE
from utils.training import set_seed, plot_history, plot_history_eval


def train(args):
    
    # args = edict(vars(args))   # edict supports: args.aug, args['aug']
    cprint(f"task_name: {args.task_name}", "magenta")

    # import dataset according to data_style
    if args.aug > 0:
        cprint("Using data augmentation (valid for 'rise' data style) ...", "magenta")
    else:
        cprint("No data augmentation ...", "magenta")
    if args.data_style == 'dp3':
        cprint("Using dp3 data style ...", "magenta")
        from dataset.sim_dataset import RobotwinDataset, collate_fn    # joint action, no data augmentation
    elif args.data_style == 'rise':
        cprint("Using rise data style ...", "magenta")
        from dataset.sim_dataset_v2 import RobotwinDataset, collate_fn   # tcp action, with point cloud augmentation
    elif args.data_style == 'real':
        cprint("Using real data ...", "magenta")
        from dataset.real_dataset import RealDataset, collate_fn         # follow rise data style, for real-world robot data
    else:
        raise ValueError(f"Unknown data_style: {args.data_style}")

    # set ckpt_dir
    ckpt_name = f"{args.task_name}-{args.train_env_setting}-{args.expert_data_num}_{args.seed}{args.extra_info}"
    ckpt_dir = pathlib.Path(os.path.join(RISE_ROOT, 'checkpoints', ckpt_name))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir = str(ckpt_dir)

    # set pose_dim
    if args.data_style in ['dp3', 'rise']:
        pose_dim_yaml = os.path.join(RISE_ROOT, 'utils/pose_dim.yaml')
        with open(pose_dim_yaml, 'r') as f:
            pose_dim = yaml.safe_load(f)[args.task_name]
        args.pose_dim = pose_dim
    elif args.data_style == 'real':
        args.pose_dim = 6 if 'pour_balls' in args.task_name else 9   # hard code
    print(f"pose_dim: {args.pose_dim}")

    # set up device
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset & dataloader
    cprint("Loading dataset ...", "magenta")
    if args.data_style in ['dp3', 'rise']:
        dataset = RobotwinDataset(
            task_name=args.task_name,
            env_setting = args.train_env_setting,
            expert_data_num = args.expert_data_num,
            split = 'train', 
            val_ratio = args.val_ratio,
            seed = args.seed,
            num_obs = args.num_obs,
            horizon = args.horizon,
            num_action = args.num_action,
            num_pose = args.num_pose,
            pose_dim = args.pose_dim,
            voxel_size = args.voxel_size,
            aug = args.aug,
        )
        do_eval = False
    else:
        # real-world data in rise data style
        dataset = RealDataset(
            task_name=args.task_name,
            split = 'train', 
            num_obs = args.num_obs,
            horizon = args.horizon,
            num_action = args.num_action,
            num_pose = args.num_pose,
            pose_dim = args.pose_dim,
            voxel_size = args.voxel_size,
            aug = args.aug,
        )
        do_eval = True
        dataset_val = RealDataset(
            task_name=args.task_name,
            split = 'val', 
            num_obs = args.num_obs,
            horizon = args.horizon,
            num_action = args.num_action,
            num_pose = args.num_pose,
            pose_dim = args.pose_dim,
            voxel_size = args.voxel_size,
            aug = args.aug,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        collate_fn = collate_fn,
        shuffle = True,
        drop_last = True,
    )
    if do_eval:
        dataloader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size = args.batch_size,
            num_workers = args.num_workers,
            collate_fn = collate_fn,
            shuffle = False,
            drop_last = False,
        )

    # policy
    cprint("Loading policy ...", "magenta")
    assert args.action_dim == dataset.action_dim, "action_dim in args must be equal to dataset.action_dim"
    policy = RISE(
        horizon = args.horizon,
        num_action = args.num_action,
        input_dim = 3, # 6,  # does not use rgb
        obs_feature_dim = args.obs_feature_dim,
        action_dim = args.action_dim,
        nheads = args.nheads,
        num_encoder_layers = args.num_encoder_layers,
        num_decoder_layers = args.num_decoder_layers,
        dropout = args.dropout,
        num_obs = args.num_obs,
    )

    if args.data_style == 'dp3':
        policy.set_normalizer(dataset.normalizer)
        cprint("DP3 style dataset: Normalizer loaded.", "magenta")
    else:
        cprint("RISE style dataset: No normalizer loaded.", "yellow")

    policy.to(device)

    # load checkpoint
    if args.resume_ckpt is not None:
        policy.load_state_dict(torch.load(args.resume_ckpt, map_location = device), strict = False)
        cprint("Checkpoint {} loaded.".format(args.resume_ckpt), "magenta")

    # optimizer and lr scheduler
    cprint("Loading optimizer and scheduler ...", "magenta")
    optimizer = torch.optim.AdamW(policy.parameters(), lr = args.lr, betas = [0.95, 0.999], weight_decay = 1e-6)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 2000,
        num_training_steps = len(dataloader) * args.num_epochs
    )
    lr_scheduler.last_epoch = len(dataloader) * (args.resume_epoch + 1) - 1

    # training
    train_history = []
    eval_history = []

    policy.train()
    for epoch in range(args.resume_epoch + 1, args.num_epochs):
        cprint("Epoch {}".format(epoch), "green") 
        optimizer.zero_grad()
        num_steps = len(dataloader)
        pbar = tqdm(dataloader)
        avg_loss = 0

        for data in pbar:
            # cloud data processing
            cloud_coords = data['input_coords_list']
            cloud_feats = data['input_feats_list']
            action_data = data['action_normalized']
            cloud_feats, cloud_coords, action_data = cloud_feats.to(device), cloud_coords.to(device), action_data.to(device)
            cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
            # forward
            loss = policy(cloud_data, action_data, batch_size = action_data.shape[0])
            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            avg_loss += loss.item()

        avg_loss = avg_loss / num_steps
        train_history.append(avg_loss)

        cprint("Train loss: {:.6f}".format(avg_loss), "green")
        # if (epoch + 1) % args.save_epochs == 0:
        #     torch.save(policy.state_dict(),
        #         os.path.join(args.ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, args.seed))
        #     )
        #     plot_history(train_history, epoch, args.ckpt_dir, args.seed)

        # do evaluation
        if do_eval:
            policy.eval()
            with torch.no_grad():
                eval_loss = 0
                num_steps_val = len(dataloader_val)
                pbar_val = tqdm(dataloader_val)
                for data_val in pbar_val:
                    # cloud data processing
                    cloud_coords_val = data_val['input_coords_list']
                    cloud_feats_val = data_val['input_feats_list']
                    action_data_val = data_val['action_normalized']
                    cloud_feats_val, cloud_coords_val, action_data_val = cloud_feats_val.to(device), cloud_coords_val.to(device), action_data_val.to(device)
                    cloud_data_val = ME.SparseTensor(cloud_feats_val, cloud_coords_val)
                    # forward
                    loss_val = policy(cloud_data_val, action_data_val, batch_size = action_data_val.shape[0])
                    eval_loss += loss_val.item()
                eval_loss = eval_loss / num_steps_val
                eval_history.append(eval_loss)
                cprint("Eval loss: {:.6f}".format(eval_loss), "green")

    torch.save(policy.state_dict(), os.path.join(args.ckpt_dir, "policy_last.ckpt"))
    plot_history(train_history, epoch, args.ckpt_dir, args.seed)
    if do_eval:
        plot_history_eval(eval_history, epoch, args.ckpt_dir, args.seed)


def arg_parse():
    parser = argparse.ArgumentParser()
    # add:
    parser.add_argument('--task_name', action = 'store', type = str, help = 'task name', required = True)
    parser.add_argument('--train_env_setting', action = 'store', type = str, help = 'train environment setting', required = False, default='demo_clean')
    parser.add_argument('--eval_env_setting', action = 'store', type = str, help = 'evaluation environment setting', required = False, default='demo_clean')
    parser.add_argument('--eval_env_seed', action = 'store', type = int, help = 'evaluation environment seed', required = False, default = 0)
    parser.add_argument('--expert_data_num', action = 'store', type = int, help = 'number of expert data', required = False, default = 50)
    parser.add_argument('--val_ratio', action = 'store', type = float, help = 'validation ratio', required = False, default = 0.02)
    parser.add_argument('--horizon', action = 'store', type = int, help = 'prediction horizon', required = False, default = 16)
    parser.add_argument('--num_pose', action = 'store', type = int, help = 'number of pose', required = False, default = 8)
    parser.add_argument('--action_dim', action = 'store', type = int, help = 'action dimension', required = False, default = 14)
    parser.add_argument('--num_obs', action = 'store', type = int, help = 'number of observation steps', required = False, default = 1)
    parser.add_argument('--extra_info', action = 'store', type = str, help = 'extra info for ckpt name', required = False, default='')  
    parser.add_argument('--data_style', action = 'store', type = str, help = 'dp3 or rise', required = False, default='dp3')
    parser.add_argument('--aug', action = 'store', type = float, help = 'probability to add point augmentation', default=-1)

    # orig
    # parser.add_argument('--data_path', action = 'store', type = str, help = 'data path', required = True)
    # parser.add_argument('--aug', action = 'store_true', help = 'whether to add 3D data augmentation')
    parser.add_argument('--aug_jitter', action = 'store_true', help = 'whether to add color jitter augmentation')
    parser.add_argument('--num_action', action = 'store', type = int, help = 'number of action steps', required = False, default = 8) # useless for RISE, since we use horizon instead.
    parser.add_argument('--voxel_size', action = 'store', type = float, help = 'voxel size', required = False, default = 0.005)
    parser.add_argument('--obs_feature_dim', action = 'store', type = int, help = 'observation feature dimension', required = False, default = 512)
    parser.add_argument('--hidden_dim', action = 'store', type = int, help = 'hidden dimension', required = False, default = 512)
    parser.add_argument('--nheads', action = 'store', type = int, help = 'number of heads', required = False, default = 8)
    parser.add_argument('--num_encoder_layers', action = 'store', type = int, help = 'number of encoder layers', required = False, default = 4)
    parser.add_argument('--num_decoder_layers', action = 'store', type = int, help = 'number of decoder layers', required = False, default = 1)
    parser.add_argument('--dim_feedforward', action = 'store', type = int, help = 'feedforward dimension', required = False, default = 2048)
    parser.add_argument('--dropout', action = 'store', type = float, help = 'dropout ratio', required = False, default = 0.1)
    # parser.add_argument('--ckpt_dir', action = 'store', type = str, help = 'checkpoint directory', required = True)
    parser.add_argument('--resume_ckpt', action = 'store', type = str, help = 'resume checkpoint file', required = False, default = None)
    parser.add_argument('--resume_epoch', action = 'store', type = int, help = 'resume from which epoch', required = False, default = -1)
    parser.add_argument('--lr', action = 'store', type = float, help = 'learning rate', required = False, default = 3e-4)
    parser.add_argument('--batch_size', action = 'store', type = int, help = 'batch size', required = False, default = 240)
    parser.add_argument('--num_epochs', action = 'store', type = int, help = 'training epochs', required = False, default = 1000)
    parser.add_argument('--save_epochs', action = 'store', type = int, help = 'saving epochs', required = False, default = 50)
    parser.add_argument('--num_workers', action = 'store', type = int, help = 'number of workers', required = False, default = 4)
    parser.add_argument('--seed', action = 'store', type = int, help = 'seed', required = False, default = 0)
    parser.add_argument('--vis_data', action = 'store_true', help = 'whether to visualize the input data and ground truth actions.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = arg_parse()
    train(args)
