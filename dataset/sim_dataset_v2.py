import sys
from pathlib import Path
RoboTwin_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(str(Path(__file__).parent.parent))  # RISE_root
from dataset.dataset_utils import load_hdf5_v2, get_flat_pose, get_val_mask
from dataset.constants import *
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform
import os
import re
import torch
import numpy as np
import MinkowskiEngine as ME
import collections.abc as container_abcs
from torch.utils.data import Dataset


# RISE data style: 
#   1. Using tcp action with data augmentation (In this script, we equivalently use endpose action).
#   2. For a training sample, we take the 1st frame as an observation and pad it with previous frames if n_obs_steps > 1.
#      This is different from the sample constructing logic in DP3 data style. We fix n_obs_steps=1 in this script.
#   3. In RISE dataloader, action and point cloud are in camera coordinate system. Here we use robotwin's world system.
#   4. Different from RISE dataloader, we empirically exclude rgb, and also crop point cloud of desktop.


DATA_DIR = os.path.join(RoboTwin_ROOT, "data")  # hard code


# vis point cloud
vis_pcd_flag = False # True # 
if vis_pcd_flag:
    import visdom
    from utils.vis_func import visualize
    vis = visdom.Visdom(env="rise_vis")


class RobotwinDataset(Dataset):
    """ Simulated Dataset. """
    def __init__(
        self, 
        task_name: str,
        env_setting = 'demo_clean',
        expert_data_num = 50,
        split = 'train', 
        val_ratio = 0.02,
        seed = 0,
        num_obs = 1,
        horizon = 20,
        num_action = 8,   # length of conditional sequence
        num_pose = 8,     # keep the same with num_action
        pose_dim = 9,
        voxel_size = 0.005,
        aug = False,
        aug_trans_min = [-0.2, -0.1, 0], # [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.1, 0],   # [0.2, 0.2, 0.2],
        aug_rot_min = [-30, -30, -30],
        aug_rot_max = [30, 30, 30],
        # unused:
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        with_cloud = False,
        vis = False
    ):
        assert num_action == num_pose
        assert split in ['train', 'val', 'all']
        val_mask = get_val_mask(n_episodes=expert_data_num, val_ratio=val_ratio, seed=seed)
        if split == 'train':
            data_mask = ~val_mask
        elif split == 'val':
            data_mask = val_mask
        elif split == 'all':
            data_mask = np.ones(expert_data_num, dtype=bool)
        
        self.split = split
        self.num_obs = num_obs
        self.horizon = horizon
        self.num_action = num_action
        self.num_pose = num_pose
        self.pose_dim = pose_dim
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.with_cloud = with_cloud
        self.vis = vis

        self.replay_buffer = self.get_replay_buffer(task_name, env_setting, expert_data_num)    # read all data
        self.set_frame_ids(expert_data_num, self.replay_buffer['episode_ends'], data_mask)
        self.action_dim = 20 #  self.replay_buffer['action'].shape[-1]

    def get_replay_buffer(self, task_name, env_setting, expert_data_num, episode_mask=None):
        replay_buffer = {}
        action_ls = []
        pointcloud_ls = []
        target_pose_ls = []
        use_arm_ls = []

        def _extract_number(filename):
            match = re.search(r'episode(\d+)\.hdf5', filename)
            if match:
                return int(match.group(1))
            return float('inf')

        self.task_data_dir = os.path.join(DATA_DIR, task_name, env_setting, 'data')
        hdf5_files = sorted(os.listdir(self.task_data_dir), key=_extract_number)
        assert len(hdf5_files) >= expert_data_num
        hdf5_files = hdf5_files[:expert_data_num]

        for i, hdf5_file in enumerate(hdf5_files):
            # if not episode_mask[i]:
            #     continue

            action, pointcloud, object_pose_ls, use_arm = load_hdf5_v2(os.path.join(self.task_data_dir, hdf5_file))
            action_ls.append(action)
            pointcloud_ls.append(pointcloud)
            use_arm_ls.append(use_arm)

            # object_pose_concat = None
            # for object_pose in object_pose_ls:
            #     if object_pose_concat is None:
            #         object_pose_concat = get_flat_pose(object_pose)
            #     else:
            #         object_pose_concat = np.concatenate([object_pose_concat, get_flat_pose(object_pose)], axis=-1)
            # assert object_pose_concat.shape[1] >= self.pose_dim
            # target_pose = object_pose_concat[:, :self.pose_dim]
            # target_pose_ls.append(target_pose)
            # 
            num_target = self.pose_dim // 9
            assert len(object_pose_ls) >= num_target
            target_pose = np.stack(object_pose_ls[:num_target], axis=1)    # [N,xx,4,4]
            target_pose_ls.append(target_pose)

        episode_ends = [action.shape[0] for action in action_ls]
        replay_buffer['episode_ends'] = np.cumsum(episode_ends)
        replay_buffer['action'] = np.concatenate(action_ls, axis=0)
        replay_buffer['pointcloud'] = np.concatenate(pointcloud_ls, axis=0)
        replay_buffer['pose'] = np.concatenate(target_pose_ls, axis=0)
        replay_buffer['use_arm'] = use_arm_ls
        return replay_buffer
        
    
    def set_frame_ids(self, data_num, episode_ends, data_mask):
        self.obs_frame_ids = []
        self.action_frame_ids = []
        self.use_arm_ls = []

        self.past_pose_frame_ids = []
        self.past_action_frame_ids = []
        cond_seq_len = self.num_action

        for i in range(data_num):
            if not data_mask[i]:
                continue
            use_arm = self.replay_buffer['use_arm'][i]

            # get frame ids
            start = 0 if i == 0 else episode_ends[i-1]
            end = episode_ends[i]
            frame_ids = list(range(start, end))

            # get samples according to num_obs and horizon
            obs_frame_ids_list = []
            action_frame_ids_list = []
            use_arm_sub_ls = []
            past_pose_ids_list = []
            past_action_ids_list = []

            for cur_idx in range(len(frame_ids) - 1):
                obs_pad_before = max(0, self.num_obs - cur_idx - 1)
                action_pad_after = max(0, self.horizon - (len(frame_ids) - 1 - cur_idx))
                frame_begin = max(0, cur_idx - self.num_obs + 1)
                frame_end = min(len(frame_ids), cur_idx + self.horizon + 1)
                obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                obs_frame_ids_list.append(obs_frame_ids)
                action_frame_ids_list.append(action_frame_ids)
                use_arm_sub_ls.append(use_arm)

                if cur_idx >= cond_seq_len:
                    past_pose_ids_list.append(frame_ids[cur_idx-cond_seq_len : cur_idx])
                    past_action_ids_list.append(frame_ids[cur_idx-cond_seq_len+1 : cur_idx+1])
                else:
                    past_pose_ids_list.append([])
                    past_action_ids_list.append([])
            
            self.obs_frame_ids += obs_frame_ids_list
            self.action_frame_ids += action_frame_ids_list
            self.use_arm_ls += use_arm_sub_ls
            self.past_pose_frame_ids += past_pose_ids_list
            self.past_action_frame_ids += past_action_ids_list
    
    def __len__(self):
        return len(self.obs_frame_ids)

    def _augmentation(self, clouds, tcps, pose, use_arm, past_tcps=None, past_pose=None):
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi                 # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)   # (4,4)
        center = clouds[-1][..., :3].mean(axis = 0)

        for i in range(len(clouds)):
            clouds[i][..., :3] -= center
            clouds[i] = apply_mat_to_pcd(clouds[i], aug_mat)
            clouds[i][..., :3] += center

        # only augment the used arm
        left_tcps  = tcps[..., :7]            # 3d_trans + 4d_quat
        right_tcps = tcps[..., 7:]
        if use_arm == 1 or use_arm == 3:      # left:1, right:2, both:3
            left_tcps[..., :3] -= center
            left_tcps = apply_mat_to_pose(left_tcps, aug_mat, rotation_rep = "quaternion")
            left_tcps[..., :3] += center
        if use_arm == 2 or use_arm == 3:
            right_tcps[..., :3] -= center
            right_tcps = apply_mat_to_pose(right_tcps, aug_mat, rotation_rep = "quaternion")
            right_tcps[..., :3] += center
        tcps = np.concatenate([left_tcps, right_tcps], axis=-1)

        seq_len = pose.shape[0]
        pose  = pose.reshape(-1, 4, 4)        # [L,xx,4,4] -> [L*xx,4,4]
        pose[..., :3, 3] -= center
        pose = apply_mat_to_pose(pose, aug_mat, rotation_rep = "matrix")
        pose[..., :3, 3] += center
        pose = pose.reshape(seq_len,-1,4,4)   # [L,xx,4,4]

        # augment the past tcps/poses
        if past_tcps is not None:
            assert past_pose is not None
            left_tcps  = past_tcps[..., :7]   # 3d_trans + 4d_quat
            right_tcps = past_tcps[..., 7:]
            if use_arm == 1 or use_arm == 3:  # left:1, right:2, both:3
                left_tcps[..., :3] -= center
                left_tcps = apply_mat_to_pose(left_tcps, aug_mat, rotation_rep = "quaternion")
                left_tcps[..., :3] += center
            if use_arm == 2 or use_arm == 3:
                right_tcps[..., :3] -= center
                right_tcps = apply_mat_to_pose(right_tcps, aug_mat, rotation_rep = "quaternion")
                right_tcps[..., :3] += center
            past_tcps = np.concatenate([left_tcps, right_tcps], axis=-1)
        
            cond_seq_len = past_pose.shape[0]
            past_pose  = past_pose.reshape(-1, 4, 4)            # [L,xx,4,4] -> [L*xx,4,4]
            past_pose[..., :3, 3] -= center
            past_pose = apply_mat_to_pose(past_pose, aug_mat, rotation_rep = "matrix")
            past_pose[..., :3, 3] += center
            past_pose = past_pose.reshape(cond_seq_len,-1,4,4)  # [L,xx,4,4]

        return clouds, tcps, pose, past_tcps, past_pose

    def _normalize_tcp(self, tcp_arr):
        ''' tcp_arr: [T, 3(left_trans) + 6(left_rot) + 3(right_trans) + 6(right_rot) + 1(left_width) + 1(right_width)]'''
        tcp_arr[:, :3] = (tcp_arr[:, :3] - TRANS_MIN_robotwin) / (TRANS_MAX_robotwin - TRANS_MIN_robotwin) * 2 - 1
        tcp_arr[:, 9:12] = (tcp_arr[:, 9:12] - TRANS_MIN_robotwin) / (TRANS_MAX_robotwin - TRANS_MIN_robotwin) * 2 - 1
        tcp_arr[:, -2:] = tcp_arr[:, -2:] * 2 - 1    # gripper width: [0,1] --> [-1,1]
        return tcp_arr
    
    def _normalize_pose(self, pose_arr):
        ''' pose_arr: [L,num_target,4,4] --> [L, 9/18/27], each 9 is [rot_6d, trans_3d]'''
        seq_len, num_target = pose_arr.shape[0], pose_arr.shape[1]
        pose_arr = get_flat_pose(pose_arr.reshape(-1,4,4)).reshape(seq_len, -1)   # [L, num_target*9]
        start = 6
        for i in range(num_target):
            pose_arr[:, start : start+3] = (pose_arr[:, start : start+3] - TRANS_MIN_robotwin) / (TRANS_MAX_robotwin - TRANS_MIN_robotwin) * 2 - 1
            start += 9
        return pose_arr

    def __getitem__(self, index):
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]
        use_arm = self.use_arm_ls[index]

        dtype = self.replay_buffer['action'].dtype
        cond_seq_len = self.num_action
        past_pose_ids = self.past_pose_frame_ids[index]
        past_action_ids = self.past_action_frame_ids[index]
        if len(past_action_ids) > 0:
            assert len(past_pose_ids) > 0
            past_action_tcps = self.replay_buffer['action'][np.array(past_action_ids)][..., :14].copy()  # (cond_seq_len, D)
            past_action_grippers = self.replay_buffer['action'][np.array(past_action_ids)][..., -2:].copy()
            past_pose = self.replay_buffer['pose'][np.array(past_pose_ids)].copy()  # (cond_seq_len, xx, 4, 4)
            action_have_past = np.ones(1, dtype=bool)
            pose_have_past = np.ones(1, dtype=bool)
        else:
            past_action_tcps = None
            past_action_grippers = None
            past_pose = None
            action_have_past = np.zeros(1, dtype=bool)
            pose_have_past = np.zeros(1, dtype=bool)
        pose = self.replay_buffer['pose'][np.array(action_frame_ids)-1].copy()  # (L, xx, 4, 4)

        # point clouds
        pcds   = self.replay_buffer['pointcloud'][np.array(obs_frame_ids)][..., :3].copy()
        colors = self.replay_buffer['pointcloud'][np.array(obs_frame_ids)][..., 3:].copy()
        # colors_norm = (colors - IMG_MEAN) / IMG_STD
        # clouds = np.concatenate([pcds, colors_norm], axis = -1)   # (L, num_pts, 6)
        clouds = pcds.copy()   # exculde rgb
        if vis_pcd_flag:
            visualize(vis, pcds[-1], win=1,  opts={'markersize': 2, 'markercolor': (255*colors[-1]).astype(np.int64)}) 

        # actions
        action_tcps = self.replay_buffer['action'][np.array(action_frame_ids)][..., :14].copy()     # (L, D)
        action_grippers = self.replay_buffer['action'][np.array(action_frame_ids)][..., -2:].copy()

        # point augmentations
        if self.split == 'train' and np.random.rand(1)[0] < self.aug:
            clouds, action_tcps, pose, past_action_tcps, past_pose = self._augmentation(clouds, action_tcps, pose, use_arm, past_action_tcps, past_pose)
        if vis_pcd_flag:
            visualize(vis,clouds[-1][:,:3], win=2,  opts={'markersize': 2, 'markercolor': (255*colors[-1]).astype(np.int64)}) 

        # rotation transformation (to 6d)
        action_tcps = np.concatenate([xyz_rot_transform(action_tcps[...,:7], from_rep = "quaternion", to_rep = "rotation_6d"),
                                      xyz_rot_transform(action_tcps[...,7:], from_rep = "quaternion", to_rep = "rotation_6d")], 
                                      axis = -1)   
        actions = np.concatenate((action_tcps, action_grippers), axis = -1)          # (L=20, dim=20)

        if past_action_tcps is not None:
            past_action_tcps = np.concatenate([xyz_rot_transform(past_action_tcps[...,:7], from_rep = "quaternion", to_rep = "rotation_6d"),
                                               xyz_rot_transform(past_action_tcps[...,7:], from_rep = "quaternion", to_rep = "rotation_6d")], 
                                        axis = -1)   
            past_actions = np.concatenate((past_action_tcps, past_action_grippers), axis = -1)  # (L=20, dim=20)
        else:
            past_actions_normalized = np.zeros(shape=(cond_seq_len, self.action_dim), dtype=dtype)
            past_pose_normalized = np.zeros(shape=(cond_seq_len, self.pose_dim), dtype=dtype)

        # normalization
        actions_normalized = self._normalize_tcp(actions.copy())
        pose_normalized = self._normalize_pose(pose.copy())  # (cond_seq_len, num_target, 4, 4) --> (cond_seq_len, num_target*9)

        if past_action_tcps is not None:
            past_actions_normalized = self._normalize_tcp(past_actions.copy())
            past_pose_normalized = self._normalize_pose(past_pose.copy())

        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            # Upd Note. Make coords contiguous.
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
            # Upd Note. API change.
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))

        # convert to torch
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()

        ret_dict = {
            'input_coords_list': input_coords_list,     # in collate_fn: (total_pt_num, 1+3), coords[:,0] is batch_idx
            'input_feats_list': input_feats_list,       # in collate_fn: (total_pt_num, 6)
            'action': actions,
            'action_normalized': actions_normalized,    # in collate_fn: (batch, L=20, action_dim=10)
            
            # 'pose': pose,
            'pose_normalized': pose_normalized,
            
            # 'pose_past': past_pose,
            'pose_past_normalized': past_pose_normalized,
            'pose_have_past': pose_have_past,
            
            # 'action_past': past_actions,
            'action_past_normalized': past_actions_normalized,
            'action_have_past': action_have_past,

            'use_arm': np.array([use_arm], dtype=int),     # (1,), int
        }
        
        return ret_dict
        

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            # if key in TO_TENSOR_KEYS:
            #     ret_dict[key] = collate_fn([d[key] for d in batch])
            # else:
            #     ret_dict[key] = [d[key] for d in batch]
            ret_dict[key] = collate_fn([d[key] for d in batch])  # replace the above 4 lines

        coords_batch = ret_dict['input_coords_list']             # list of array, list_length = batch * num_obs
        feats_batch = ret_dict['input_feats_list']
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)  # (total_num_pts, 4), (total_num_pts, 6)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


if __name__ == '__main__':
    task_name = 'beat_block_hammer'
    dataset = RobotwinDataset(task_name, num_obs=1, horizon=10, aug=True)
    for i in range(len(dataset)):
        sample = dataset[i]
    print('test done')
