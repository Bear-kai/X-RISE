import sys, os
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(current_file_path))
from dataset.dataset_utils import load_hdf5, get_flat_pose, get_val_mask, create_indices, LinearNormalizer

import pathlib
RoboTwin_ROOT = str(pathlib.Path(__file__).resolve().parent.parent.parent.parent)

import re
import torch
import numpy as np
import MinkowskiEngine as ME
import collections.abc as container_abcs
from torch.utils.data import Dataset

# DP3 data style: 
#   1. Using joint action without data augmentation. 
#   2. For a training sample, we use the first n_obs_steps frames as observations. Therefore, 
#      only pred[(n_obs_steps-1):] is used for execution. For (X-)RISE, we fix n_obs_steps=1.


DATA_DIR = os.path.join(RoboTwin_ROOT, "data")  # hard code


# vis point cloud
vis_pcd_flag = False #  True  #  
if vis_pcd_flag:
    from PIL import Image
    import visdom
    from utils.vis_func import visualize
    vis = visdom.Visdom(env="robotwin_vis")


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
        # unused:
        aug = False,
        aug_trans_min = [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.2, 0.2],
        aug_rot_min = [-30, -30, -30],
        aug_rot_max = [30, 30, 30],
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        with_cloud = False,
        vis = False
    ):  
        assert num_obs == 1
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
        self.indices, self.use_arm_arr = create_indices(self.replay_buffer['episode_ends'], 
                                                        self.replay_buffer['use_arm'],
                                                        horizon, data_mask, 
                                                        pad_before=num_obs-1, pad_after=horizon-1)
        self.normalizer = self.get_normalizer()
        self.action_dim = self.replay_buffer['action'].shape[-1]

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

            action, pointcloud, object_pose_ls, use_arm = load_hdf5(os.path.join(self.task_data_dir, hdf5_file))
            action_ls.append(action)
            pointcloud_ls.append(pointcloud)
            use_arm_ls.append(use_arm)

            object_pose_concat = None
            for object_pose in object_pose_ls:
                if object_pose_concat is None:
                    object_pose_concat = get_flat_pose(object_pose)
                else:
                    object_pose_concat = np.concatenate([object_pose_concat, get_flat_pose(object_pose)], axis=-1)
            assert object_pose_concat.shape[1] >= self.pose_dim
            target_pose = object_pose_concat[:, :self.pose_dim]
            target_pose_ls.append(target_pose)

        episode_ends = [action.shape[0] for action in action_ls]
        replay_buffer['episode_ends'] = np.cumsum(episode_ends)
        replay_buffer['action'] = np.concatenate(action_ls, axis=0)
        replay_buffer['pointcloud'] = np.concatenate(pointcloud_ls, axis=0)
        replay_buffer['pose'] = np.concatenate(target_pose_ls, axis=0)
        replay_buffer['use_arm'] = use_arm_ls
        return replay_buffer
        
    def __len__(self):
        return len(self.indices)

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            # "pointcloud": self.replay_buffer["pointcloud"],
            'pose': self.replay_buffer["pose"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    
    def _get_sample_in_buffer(self, index):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, start_idx, end_idx = self.indices[index]
        result = dict()

        result['use_arm'] = self.use_arm_arr[index]

        for key in ['pointcloud', 'pose']:
            input_arr = self.replay_buffer[key]
            sample = input_arr[buffer_start_idx : buffer_end_idx]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.horizon):
                data = np.zeros(shape=(self.horizon,) + input_arr.shape[1:], dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.horizon:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data

        action = self.replay_buffer['action']
        sample = action[buffer_start_idx+1 : min(buffer_end_idx+1, end_idx)]
        if len(sample) < self.horizon:
            data = np.zeros(shape=(self.horizon,) + action.shape[1:], dtype=action.dtype)
            if buffer_start_idx == start_idx:
                data[-len(sample):] = sample
                data[:-len(sample)] = sample[0]
            elif buffer_end_idx == end_idx:
                data[:len(sample)] = sample
                data[len(sample):] = sample[-1]
        else:
            data = sample
        result['action'] = data

        cond_seq_len = self.num_pose
        buffer_start_idx_offset = buffer_start_idx + (self.num_obs - 1)
        buffer_start_idx_offset = min(buffer_start_idx_offset, end_idx - 1)
        pose = self.replay_buffer['pose']
        if (buffer_start_idx_offset - start_idx) < cond_seq_len:
            result['pose_past'] = np.zeros(shape=(cond_seq_len,) + pose.shape[1:], dtype=pose.dtype)
            result['pose_have_past'] = np.zeros(1, dtype=bool)
        else:
            result['pose_past'] = pose[(buffer_start_idx_offset - cond_seq_len) : buffer_start_idx_offset]
            result['pose_have_past'] = np.ones(1, dtype=bool)

        if (buffer_start_idx_offset - start_idx) < cond_seq_len:
            result['action_past'] = np.zeros(shape=(cond_seq_len,) + action.shape[1:], dtype=action.dtype)
            result['action_have_past'] = np.zeros(1, dtype=bool)
        else:
            result['action_past'] = action[(buffer_start_idx_offset - cond_seq_len)+1 : buffer_start_idx_offset+1]
            result['action_have_past'] = np.ones(1, dtype=bool)

        return result

    def __getitem__(self, index):
        
        sample = self._get_sample_in_buffer(index)

        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for i in range(self.num_obs):
            cloud = sample['pointcloud'][i][:,:3]   # only xyz, no rgb
            # Upd Note. Make coords contiguous.
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
            # Upd Note. API change.
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))

            if vis_pcd_flag:
                visualize(vis, cloud[:,:3], win=1, opts={'markersize': 2, 'markercolor': (cloud[:,3:]*255).astype(np.int64)})
            
        # convert to torch
        actions = torch.from_numpy(sample['action']).float()
        actions_normalized = self.normalizer['action'].normalize(actions)

        pose = torch.from_numpy(sample['pose']).float()
        pose_normalized = self.normalizer['pose'].normalize(pose)

        pose_past = torch.from_numpy(sample['pose_past']).float()
        pose_past_normalized = self.normalizer['pose'].normalize(pose_past)
        pose_have_past = torch.from_numpy(sample['pose_have_past'])

        action_past = torch.from_numpy(sample['action_past']).float()
        action_past_normalized = self.normalizer['action'].normalize(action_past)
        action_have_past = torch.from_numpy(sample['action_have_past'])

        use_arm = torch.from_numpy(sample['use_arm']).int()
        
        ret_dict = {
            'input_coords_list': input_coords_list,     # in collate_fn: (total_pt_num, 1+3), coords[:,0] is batch_idx
            'input_feats_list': input_feats_list,       # in collate_fn: (total_pt_num, 6)
            'action': actions,
            'action_normalized': actions_normalized,    # in collate_fn: (batch, L=20, action_dim=10)
            
            # 'pose': pose,
            'pose_normalized': pose_normalized,
            
            # 'pose_past': pose_past,
            'pose_past_normalized': pose_past_normalized,
            'pose_have_past': pose_have_past,
            
            # 'action_past': action_past,
            'action_past_normalized': action_past_normalized,
            'action_have_past': action_have_past,

            'use_arm': use_arm,     # (1,), int
        }
        
        return ret_dict
        

# TO_TENSOR_KEYS = ['input_coords_list', 'input_feats_list', 'action', 'action_normalized']
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
    dataset = RobotwinDataset(task_name, num_obs=3, horizon=10, num_action=5, num_pose=5)
    for i in range(len(dataset)):
        sample = dataset[i]
    print('test done')