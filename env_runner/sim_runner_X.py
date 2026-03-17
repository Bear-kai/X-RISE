import numpy as np
import torch
from queue import deque
import MinkowskiEngine as ME

from utils.constants import *
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion


# vis point cloud
vis_pcd_flag = False #  True  #  
if vis_pcd_flag:
    from PIL import Image
    from utils.vis_func import visualize
    import visdom
    vis = visdom.Visdom(env="robotwin_vis")


def unnormalize_and_transform(action):     # for rise data style
    """ 
    input:  [T, 3(left_trans) + 6(left_rot) + 3(right_trans) + 6(right_rot) + 1(left_width) + 1(right_width)] 
    return: [T, 3(left_trans) + 4(left_rot) + 1(left_width) + 3(right_trans) + 6(right_rot) + 1(right_width)], 
    """
    # unnormalize
    TRANS_MAX = torch.from_numpy(TRANS_MAX_robotwin).to(action.device)
    TRANS_MIN = torch.from_numpy(TRANS_MIN_robotwin).to(action.device)
    
    left_trans  = (action[:, :3] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    right_trans = (action[:, 9:12] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    left_width  = (action[:, -2] + 1) / 2.0
    right_width = (action[:, -1] + 1) / 2.0
    
    # transform
    left_rot_quat  = matrix_to_quaternion(rotation_6d_to_matrix(action[:, 3:9]))
    right_rot_quat = matrix_to_quaternion(rotation_6d_to_matrix(action[:, 12:18]))

    action = torch.cat([left_trans, left_rot_quat, left_width.unsqueeze(1), 
                        right_trans, right_rot_quat, right_width.unsqueeze(1)], dim=1)

    return action


class BaseRunner:

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy):
        raise NotImplementedError()


class RobotRunner(BaseRunner):

    def __init__(
        self,
        n_obs_steps=8,
        n_action_steps=8,
        action_dim=14,
        n_pose_steps=8,
        pose_dim=9,
        device=None,
        dtype=None,
        voxel_size=0.005,
        output_dir=None,
    ):
        super().__init__(output_dir)
        self.n_obs_steps = n_obs_steps
        self.voxel_size = voxel_size
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.n_pose_steps = n_pose_steps
        self.pose_dim = pose_dim
        self.device = device     
        self.dtype = dtype
        self.obs = deque(maxlen=n_obs_steps + 1)

        self.action_past = torch.zeros((1, n_action_steps, action_dim)).to(device=device, dtype=dtype)
        self.pose_past   = torch.zeros((1, n_pose_steps ,  pose_dim)).to(device=device, dtype=dtype)
        self.action_have_past = torch.ones((1,1)).to(device=device, dtype=torch.bool)    # fix true
        self.pose_have_past   = torch.ones((1,1)).to(device=device, dtype=torch.bool)    # fix true

    def stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert len(self.obs) > 0, "no observation is recorded, please update obs first"

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs([obs[key] for obs in self.obs], self.n_obs_steps)

        return result

    def get_action(self, policy, observaton=None) -> bool:
        if observaton is not None:
            self.obs.append(observaton)  # update
        obs = self.get_n_steps_obs()

        # create obs dict
        np_obs_dict = dict(obs)
        # device transfer
        # obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        # run policy
        with torch.no_grad():
            # make voxel input
            input_coords_list = []
            input_feats_list = []
            for i in range(self.n_obs_steps):
                cloud = np_obs_dict["pointcloud"][i]
                # Upd Note. Make coords contiguous.
                coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
                # Upd Note. API change.
                input_coords_list.append(coords)
                input_feats_list.append(cloud.astype(np.float32))

                if vis_pcd_flag:
                    visualize(vis, cloud[:,:3], win=1, opts={'markersize': 2, 'markercolor': (cloud[:,3:]*255).astype(np.int64)})
            
            cloud_coords, cloud_feats = ME.utils.sparse_collate(input_coords_list, input_feats_list)
            cloud_data = ME.SparseTensor(cloud_feats.to(self.device), cloud_coords.to(self.device))

            batch = {'cloud': cloud_data,
                    'action_past_normalized': self.action_past,
                    'pose_past_normalized': self.pose_past,
                    'pose_have_past':    self.pose_have_past,
                    'action_have_past':  self.action_have_past,
                    }

            # predict
            pred_raw_action, pred_raw_pose = policy(batch, actions=None, poses=None, batch_size=1)  # .squeeze(0)
            
            self.pose_past   = pred_raw_pose
            self.action_past = pred_raw_action
            if policy.have_normalizer:
                # dp3 data style
                action = policy.normalizer['action'].unnormalize(pred_raw_action.squeeze(0)).cpu().numpy()
            else:
                # rise data style
                action = unnormalize_and_transform(pred_raw_action.squeeze(0)).cpu().numpy()
            
        return action

    def run(self, policy):
        pass


if __name__ == "__main__":
    test = RobotRunner("./")
    print("ready")
