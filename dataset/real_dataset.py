import sys
from pathlib import Path
RoboTwin_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(str(Path(__file__).parent.parent))  # RISE_root
from dataset.dataset_utils import get_flat_pose, get_flat_pose_v2
from dataset.constants import *
from dataset.projector import Projector
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform
import os
import json
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import collections.abc as container_abcs
from torch.utils.data import Dataset
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools


# Follow rise data style. 


DATA_DIR = "/home/robot_data"   # hard code

# vis point cloud
vis_pcd_flag = False # True # 
if vis_pcd_flag:
    import pytorch3d.ops as torch3d_ops
    import visdom
    from utils.vis_func import visualize, gen_obj_frame_colored_pcd
    vis = visdom.Visdom(env="rise_vis")


def fps(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

# ------------------- for multi-process
def load_point_cloud(colors, depths, cam_id, voxel_size):
    h, w = depths.shape
    sx = 1 # self.target_img_size[0] / self.src_img_size[0]
    sy = 1 # self.target_img_size[1] / self.src_img_size[1]
    fx, fy = INTRINSICS[cam_id][0, 0] * sx, INTRINSICS[cam_id][1, 1] * sy
    cx, cy = INTRINSICS[cam_id][0, 2] * sx, INTRINSICS[cam_id][1, 2] * sy
    scale = 1000. if 'f' not in cam_id else 4000.
    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, scale, convert_rgb_to_intensity = False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points, dtype=np.float32)
    colors = np.array(cloud.colors, dtype=np.float32)

    x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
    y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
    z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
    mask = (x_mask & y_mask & z_mask)
    points = points[mask]
    colors = colors[mask]

    return points, colors

def process_demo(i, data_path, all_demos, cam_ids, load_point_cloud_func, voxel_size):
    demo_path = os.path.join(data_path, all_demos[i])
    print(f'  Loading: {demo_path}')
    
    local_clouds = []
    local_tcps = []
    local_gripper_widths = []
    local_poses = []
    
    projector_cache = {}
    def get_projector(demo_path):
        timestamp_path = os.path.join(demo_path, 'timestamp.txt')
        with open(timestamp_path, 'r') as f:
            timestamp = f.readline().rstrip()
        if timestamp not in projector_cache:
            projector_cache[timestamp] = Projector(os.path.join(data_path, "..", timestamp))
        return projector_cache[timestamp]
    
    projector = get_projector(demo_path)
    
    for cam_id in cam_ids:
        cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))
        if not os.path.exists(cam_path):
            continue

        meta_path = os.path.join(demo_path, "metadata.json")
        if not os.path.exists(meta_path):
            meta = {'finish_time': float('inf')}
        else:
            with open(meta_path, "r") as f:
                meta = json.load(f)

        frame_ids = [
            int(os.path.splitext(x)[0]) 
            for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
            if int(os.path.splitext(x)[0]) <= meta["finish_time"]
        ]
        
        for frame_id in frame_ids:
            rgb_path   = os.path.join(cam_path, "color", "{}.png".format(frame_id))
            depth_path = rgb_path.replace("color", "depth")
            tcp_path   = rgb_path.replace("color", "tcp").replace("png", "npy")
            gripper_path = rgb_path.replace("color", "gripper_command").replace("png", "npy")
            pose_path  = rgb_path.replace("color", "pose").replace("png", "txt")

            color = np.array(Image.open(rgb_path), dtype=np.uint8)
            depth = np.array(Image.open(depth_path), dtype=np.float32)
            
            # generate point cloud
            points, colors = load_point_cloud_func(color, depth, cam_id, voxel_size)

            # crop point cloud: 1) transform to base frame; 2) manually crop the desktop points; 3) back to camera frame. 
            # Note that RISE uses an unnormal expression: For example, cam_to_base actually means the transformation from base to cam.
            points_base = np.linalg.inv(projector.cam_to_base[cam_id]) @ np.concatenate([points, np.ones_like(points[:, :1])], axis=-1).T  # (4, num_pts)
            adjust_degY = 2.5
            adjust_degX = -0.5
            manual_rotY_mat = np.array([[np.cos(np.deg2rad(adjust_degY)), 0, np.sin(np.deg2rad(adjust_degY))],
                                        [0, 1, 0],
                                        [-np.sin(np.deg2rad(adjust_degY)), 0, np.cos(np.deg2rad(adjust_degY))]], 
                                        dtype=np.float32)
            manual_rotX_mat = np.array([[1, 0, 0],
                                        [0, np.cos(np.deg2rad(adjust_degX)), -np.sin(np.deg2rad(adjust_degX))],
                                        [0, np.sin(np.deg2rad(adjust_degX)), np.cos(np.deg2rad(adjust_degX))]], 
                                        dtype=np.float32)
            points_base = manual_rotX_mat @ manual_rotY_mat @ points_base[:3]  # (3, num_pts)
            # points = points_base[:, points_base[2] >= 0.005].T
            points = points[points_base[2] >= 0.005]
            colors = colors[points_base[2] >= 0.005]
    
            cloud = np.concatenate([points, colors], axis=-1)
            local_clouds.append(cloud)

            tcp = np.load(tcp_path)[:7].astype(np.float32)
            projected_tcp = projector.project_tcp_to_camera_coord(tcp, cam_id)
            gripper_width = np.load(gripper_path)[0]
            gripper_width = decode_gripper_width(gripper_width)
            local_tcps.append(projected_tcp)
            local_gripper_widths.append(gripper_width)

            pose = np.loadtxt(pose_path, dtype=np.float32)
            pose = pose.reshape(1,4,4) if len(pose.shape)==2 else pose
            local_poses.append(pose)
    
    return i, local_clouds, local_tcps, local_gripper_widths, local_poses, len(local_clouds)

# ------------------- 

class RealDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self, 
        task_name: str,
        split = 'train', 
        cam_ids = ['104122063550'],
        val_sample_id = [9,29,49],
        num_obs = 1,
        horizon = 20,
        num_action = 8,   # length of conditional sequence
        num_pose = 8,     # keep the same with num_action
        pose_dim = 9,
        action_dim = 10,
        voxel_size = 0.005,
        aug = False,
        aug_trans_min = [-0.2, -0.2, 0], # [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.2, 0],   # [0.2, 0.2, 0.2],
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
        self.task_name = task_name
        self.split = split
        self.cam_ids = cam_ids
        self.val_sample_id = val_sample_id
        self.num_obs = num_obs
        self.horizon = horizon
        self.num_action = num_action
        self.num_pose = num_pose
        self.pose_dim = pose_dim
        self.action_dim = action_dim
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

        # get all demo paths
        self.data_path = os.path.join(DATA_DIR, task_name, 'train')
        self.all_demos = sorted(os.listdir(self.data_path))
        self.num_demos = len(self.all_demos)

        # create color jitter
        if self.split == 'train' and self.aug_jitter:
            jitter = T.ColorJitter(
                brightness = self.aug_jitter_params[0],
                contrast = self.aug_jitter_params[1],
                saturation = self.aug_jitter_params[2],
                hue = self.aug_jitter_params[3]
            )
            self.jitter = T.RandomApply([jitter], p = self.aug_jitter_prob)

        # pre-set ids list
        self.obs_frame_ids = []
        self.action_frame_ids = []
        self.past_pose_frame_ids = []
        self.past_action_frame_ids = []
        self.projectors = {}

        # self.src_img_size = (1280, 720)     # (w,h)
        # self.target_img_size = (1280, 720)  # (640, 480)

        print('Start loading replay buffer:')
        start_time = time()
        # self.replay_buffer = self.get_replay_buffer()                   # load 9 episodes，cost 279s
        self.replay_buffer = self.get_replay_buffer_parallel_process()    # load 9 episodes with 10 processes，cost 42s
        print(f'Loading replay buffer costs {time() - start_time:.2f} seconds.')
        self.set_frame_ids()

    def get_data_mask(self):
        val_mask = np.zeros(self.num_demos, dtype=bool)
        val_mask[self.val_sample_id] = True
        if self.split == 'train':
            data_mask = ~val_mask
        elif self.split == 'val':
            data_mask = val_mask
        elif self.split == 'all':
            data_mask = np.ones(self.num_demos, dtype=bool)
        return data_mask

    def load_point_cloud(self, colors, depths, cam_id):
        h, w = depths.shape
        sx = 1 # self.target_img_size[0] / self.src_img_size[0]
        sy = 1 # self.target_img_size[1] / self.src_img_size[1]
        fx, fy = INTRINSICS[cam_id][0, 0] * sx, INTRINSICS[cam_id][1, 1] * sy
        cx, cy = INTRINSICS[cam_id][0, 2] * sx, INTRINSICS[cam_id][1, 2] * sy
        scale = 1000. if 'f' not in cam_id else 4000.
        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, scale, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        points = np.array(cloud.points, dtype=np.float32)
        colors = np.array(cloud.colors, dtype=np.float32)

        x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
        y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
        z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
        mask = (x_mask & y_mask & z_mask)
        points = points[mask]
        colors = colors[mask]

        return points, colors
    
    def get_projector(self, demo_path):
        timestamp_path = os.path.join(demo_path, 'timestamp.txt')
        with open(timestamp_path, 'r') as f:
            timestamp = f.readline().rstrip()
        if timestamp not in self.projectors:
            # create projector cache
            self.projectors[timestamp] = Projector(os.path.join(self.data_path, "..", timestamp))
        projector = self.projectors[timestamp]

        return projector

    def get_replay_buffer(self):
        data_mask = self.get_data_mask()

        cloud_ls = []
        tcp_ls = []
        gripper_width_ls = []
        pose_ls = []
        episode_ends = []

        for i in range(self.num_demos):
            # if i not in [0]: # range(10): #  
            #     continue

            if not data_mask[i]:
                continue
            demo_path = os.path.join(self.data_path, self.all_demos[i])
            print(f'  Loading: {demo_path}')

            # load camera projector
            projector = self.get_projector(demo_path)
            
            for cam_id in self.cam_ids:
                # path
                cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))
                if not os.path.exists(cam_path):
                    continue
                
                # metadata
                meta_path = os.path.join(demo_path, "metadata.json")
                if not os.path.exists(meta_path):
                    meta = {'finish_time': float('inf')}
                else:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                        
                # get frame ids
                frame_ids = [
                    int(os.path.splitext(x)[0]) 
                    for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
                    if int(os.path.splitext(x)[0]) <= meta["finish_time"]
                ]
                
                for frame_id in frame_ids:
                    # get paths
                    rgb_path   = os.path.join(cam_path, "color", "{}.png".format(frame_id))
                    depth_path = rgb_path.replace("color", "depth")
                    tcp_path   = rgb_path.replace("color", "tcp").replace("png", "npy")
                    gripper_path = rgb_path.replace("color", "gripper_command").replace("png", "npy")
                    pose_path  = rgb_path.replace("color", "pose").replace("png", "txt")

                    # read rgbd images
                    color = np.array(Image.open(rgb_path), dtype=np.uint8)     # Image.open(rgb_path).resize(self.target_img_size)
                    # if self.split == 'train' and self.aug_jitter:
                    #     color = self.jitter(color)  # out: uint8
                    depth = np.array(Image.open(depth_path), dtype=np.float32) # Image.open(depth_path).resize(self.target_img_size, resample=Image.NEAREST)
                    
                    # generate point cloud
                    points, colors = self.load_point_cloud(color, depth, cam_id)  # out: float32
                    
                    # crop point cloud: 1) transform to base frame; 2) manually crop the desktop points; 3) back to camera frame. 
                    # Note that RISE uses an unnormal expression: For example, cam_to_base actually means the transformation from base to cam.
                    points_base = np.linalg.inv(projector.cam_to_base[cam_id]) @ np.concatenate([points, np.ones_like(points[:, :1])], axis=-1).T  # (4, num_pts)
                    adjust_degY = 2.5
                    adjust_degX = -0.5
                    manual_rotY_mat = np.array([[np.cos(np.deg2rad(adjust_degY)), 0, np.sin(np.deg2rad(adjust_degY))],
                                               [0, 1, 0],
                                               [-np.sin(np.deg2rad(adjust_degY)), 0, np.cos(np.deg2rad(adjust_degY))]], 
                                               dtype=np.float32)
                    manual_rotX_mat = np.array([[1, 0, 0],
                                               [0, np.cos(np.deg2rad(adjust_degX)), -np.sin(np.deg2rad(adjust_degX))],
                                               [0, np.sin(np.deg2rad(adjust_degX)), np.cos(np.deg2rad(adjust_degX))]], 
                                               dtype=np.float32)
                    points_base = manual_rotX_mat @ manual_rotY_mat @ points_base[:3]  # (3, num_pts)
                    # points = points_base[:, points_base[2] >= 0.005].T
                    points = points[points_base[2] >= 0.005]
                    colors = colors[points_base[2] >= 0.005]

                    if vis_pcd_flag:
                        markercolor = (255*colors).astype(np.int64)
                        pcd_vis, index_vis = fps(points, 1500)
                        index_vis = index_vis.detach().cpu().numpy()[0]
                        color_vis = markercolor[index_vis]
                        visualize(vis, pcd_vis, win=1,  opts={'markersize': 2, 'markercolor': color_vis}) 

                    # apply imagenet normalization
                    # colors = (colors - IMG_MEAN) / IMG_STD
                    cloud = np.concatenate([points, colors], axis = -1)   # (num_pts, 6)
                    cloud_ls.append(cloud)

                    # get actions
                    tcp = np.load(tcp_path)[:7].astype(np.float32)
                    projected_tcp = projector.project_tcp_to_camera_coord(tcp, cam_id)
                    gripper_width = np.load(gripper_path)[0]                # 0~1000
                    gripper_width = decode_gripper_width(gripper_width)     # 0~0.095m
                    tcp_ls.append(projected_tcp)
                    gripper_width_ls.append(gripper_width)

                    # get pose
                    pose = np.loadtxt(pose_path, dtype=np.float32)
                    pose = pose.reshape(1,4,4) if len(pose.shape)==2 else pose
                    pose_ls.append(pose)
                
                # record episode ends
                episode_ends.append(len(cloud_ls))
          
        # store in replay buffer
        replay_buffer = {
            "episode_ends": np.array(episode_ends),
            "cloud_ls": cloud_ls,
            "poses": np.stack(pose_ls, axis = 0),
            "tcps": (np.stack(tcp_ls, axis = 0)).astype(np.float32),
            "gripper_widths": (np.stack(gripper_width_ls, axis = 0)[:, None]).astype(np.float32),
        }
        return replay_buffer

    def get_replay_buffer_parallel_process(self, num_workers=10):
        data_mask = self.get_data_mask()
        valid_demo_indices = [i for i in range(self.num_demos) if data_mask[i]]
        # valid_demo_indices = [i for i in valid_demo_indices if i in range(10)]
        
        func = functools.partial(
            process_demo,
            data_path=self.data_path,
            all_demos=self.all_demos,
            cam_ids=self.cam_ids,
            load_point_cloud_func=load_point_cloud,
            voxel_size=self.voxel_size
        )
        
        results = [None] * len(valid_demo_indices)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(func, idx): idx for idx in valid_demo_indices}
            for future in as_completed(future_to_idx.keys()):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    pos = valid_demo_indices.index(idx)
                    results[pos] = result
                except Exception as exc:
                    print(f'Demo {idx} generated an exception: {exc}')
        
        cloud_ls, tcp_ls, gripper_width_ls, pose_ls, episode_ends = [], [], [], [], []
        global_start = 0
        for res in results:
            if res is not None:
                i, clouds, tcps, grippers, poses, n_frames = res
                cloud_ls.extend(clouds)
                tcp_ls.extend(tcps)
                gripper_width_ls.extend(grippers)
                pose_ls.extend(poses)
                global_start += n_frames
                episode_ends.append(global_start)
        
        return {
            "episode_ends": np.array(episode_ends),
            "cloud_ls": cloud_ls,
            "poses": np.stack(pose_ls, axis=0),
            "tcps": np.stack(tcp_ls, axis=0).astype(np.float32),
            "gripper_widths": np.stack(gripper_width_ls, axis=0)[:, None].astype(np.float32),
        }

    def set_frame_ids(self):    
        cond_seq_len = self.num_action
        episode_ends = self.replay_buffer['episode_ends']
        data_num = len(episode_ends)

        for i in range(data_num):
            # get frame ids
            start = 0 if i == 0 else episode_ends[i-1]
            end = episode_ends[i]
            frame_ids = list(range(start, end))

            # get samples according to num_obs and horizon
            obs_frame_ids_list = []
            action_frame_ids_list = []
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
                
                if cur_idx >= cond_seq_len:
                    past_pose_ids_list.append(frame_ids[cur_idx-cond_seq_len : cur_idx])
                    past_action_ids_list.append(frame_ids[cur_idx-cond_seq_len+1 : cur_idx+1])
                else:
                    past_pose_ids_list.append([])
                    past_action_ids_list.append([])
            
            self.obs_frame_ids += obs_frame_ids_list
            self.action_frame_ids += action_frame_ids_list
            self.past_pose_frame_ids += past_pose_ids_list
            self.past_action_frame_ids += past_action_ids_list

        return 0
    
    def __len__(self):
        return len(self.obs_frame_ids)

    def _augmentation(self, clouds, tcps, pose, past_tcps=None, past_pose=None):
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi                 # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)   # (4,4)
        center = clouds[-1][..., :3].mean(axis = 0)

        # augment point cloud
        for i in range(len(clouds)):
            clouds[i][..., :3] -= center
            clouds[i] = apply_mat_to_pcd(clouds[i], aug_mat)
            clouds[i][..., :3] += center

        # augment action
        tcps[..., :3] -= center              # 3d_trans + 4d_quat
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep = "quaternion")
        tcps[..., :3] += center

        # augment pose
        seq_len = pose.shape[0]
        pose  = pose.reshape(-1, 4, 4)       # [L,xx,4,4] -> [L*xx,4,4]
        pose[..., :3, 3] -= center
        pose = apply_mat_to_pose(pose, aug_mat, rotation_rep = "matrix")
        pose[..., :3, 3] += center
        pose = pose.reshape(seq_len,-1,4,4)  # [L,xx,4,4]

        # augment past action and pose
        if past_tcps is not None:
            assert past_pose is not None

            past_tcps[..., :3] -= center
            past_tcps = apply_mat_to_pose(past_tcps, aug_mat, rotation_rep = "quaternion")
            past_tcps[..., :3] += center

            cond_seq_len = past_pose.shape[0]
            past_pose  = past_pose.reshape(-1, 4, 4)            # [L,xx,4,4] -> [L*xx,4,4]
            past_pose[..., :3, 3] -= center
            past_pose = apply_mat_to_pose(past_pose, aug_mat, rotation_rep = "matrix")
            past_pose[..., :3, 3] += center
            past_pose = past_pose.reshape(cond_seq_len,-1,4,4)  # [L,xx,4,4]

        return clouds, tcps, pose, past_tcps, past_pose

    def _normalize_tcp(self, tcp_arr):  # normalize into [-1,1]
        ''' tcp_arr: [T, 3(trans) + 6(rot) + 1(gripper_width)]'''
        tcp_arr[:, :3] = (tcp_arr[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_arr[:, -1:] = tcp_arr[:, -1:] / MAX_GRIPPER_WIDTH * 2 - 1  
        return tcp_arr
    
    def _normalize_pose(self, pose_arr):
        ''' pose_arr: [L,num_target,4,4] --> [L, 9/18/27], each 9 is [rot_6d, trans_3d]'''
        seq_len, num_target = pose_arr.shape[0], pose_arr.shape[1]
        pose_arr = get_flat_pose(pose_arr.reshape(-1,4,4)).reshape(seq_len, -1)   # [L,num_target*9]
        start = 6
        for i in range(num_target):
            pose_arr[:, start : start+3] = (pose_arr[:, start : start+3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
            start += 9
        return pose_arr
    
    def _normalize_pose_v2(self, pose_arr, axis=1):
        ''' pose_arr: [L,num_target,4,4] --> [L, 6/12/18], each 6 is [rot_3d, trans_3d],
            axis: which axis to choose, 0/1/2 corresponds to x/y/z axis
        '''
        seq_len, num_target = pose_arr.shape[0], pose_arr.shape[1]
        pose_arr = get_flat_pose_v2(pose_arr.reshape(-1,4,4), axis).reshape(seq_len, -1)   # [L,num_target*6]
        start = 3
        for i in range(num_target):
            pose_arr[:, start : start+3] = (pose_arr[:, start : start+3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
            start += 6
        return pose_arr

    def __getitem__(self, index):
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]

        cond_seq_len = self.num_action
        past_pose_ids = self.past_pose_frame_ids[index]
        past_action_ids = self.past_action_frame_ids[index]
        if len(past_action_ids) > 0:
            assert len(past_pose_ids) > 0
            past_action_tcps = self.replay_buffer['tcps'][np.array(past_action_ids)].copy()  # (cond_seq_len, 7)
            past_action_grippers = self.replay_buffer['gripper_widths'][np.array(past_action_ids)].copy()
            past_pose = self.replay_buffer['poses'][np.array(past_pose_ids)].copy()          # (cond_seq_len, 1, 4, 4)
            action_have_past = np.ones(1, dtype=bool)
            pose_have_past = np.ones(1, dtype=bool)
        else:
            past_action_tcps = None
            past_action_grippers = None
            past_pose = None
            action_have_past = np.zeros(1, dtype=bool)
            pose_have_past = np.zeros(1, dtype=bool)
        pose = self.replay_buffer['poses'][np.array(action_frame_ids)-1].copy()        # (L, 1, 4, 4)

        # point clouds
        clouds = []
        for obs_frame_id in obs_frame_ids:
            clouds.append(self.replay_buffer['cloud_ls'][obs_frame_id][:,:3].copy())   # exclude rgb
        if vis_pcd_flag:
            # vis cloud
            markercolor = self.replay_buffer['cloud_ls'][obs_frame_ids[-1]][:,3:].copy()
            markercolor = (255*markercolor).astype(np.int64)
            pcd_vis, index_vis = fps(clouds[-1][:,:3], 3000)
            index_vis = index_vis.detach().cpu().numpy()[0]
            color_vis = markercolor[index_vis]
            
            # # vis object pose in cam
            # obj_frame_pcd, obj_frame_color = gen_obj_frame_colored_pcd(pose[0,0,:3,3], pose[0,0,:3,:3], scale=0.2)
            # # vis tcp pose in cam
            # tcp_pose = self.replay_buffer['tcps'][obs_frame_ids[-1]].copy()
            # tcp_pose = xyz_rot_transform(tcp_pose, from_rep = "quaternion", to_rep = "matrix")
            # tcp_frame_pcd, tcp_frame_color = gen_obj_frame_colored_pcd(tcp_pose[:3,3], tcp_pose[:3,:3], scale=0.22)
            # # vis base coordinate frame in cam
            # projector = self.projectors["1770283618030"]       # hang_mug & arrange_truck: 1770283618030; pour_balls: 1752812316696
            # base_pose = projector.cam_to_base["104122063550"]  # actually base2cam
            # base_frame_pcd, base_frame_color = gen_obj_frame_colored_pcd(base_pose[:3,3], base_pose[:3,:3], scale=0.25)
            # # concate
            # pcd_vis = np.concatenate([pcd_vis, obj_frame_pcd, tcp_frame_pcd, base_frame_pcd])
            # color_vis = np.concatenate([color_vis, obj_frame_color, tcp_frame_color, base_frame_color])
            
            visualize(vis, pcd_vis, win=1,  opts={'markersize': 2, 'markercolor': color_vis}) 

        # actions
        action_tcps = self.replay_buffer['tcps'][np.array(action_frame_ids)].copy()                # (L, 7)
        action_grippers = self.replay_buffer['gripper_widths'][np.array(action_frame_ids)].copy()  # (L, 1)

        # point augmentations
        if self.split == 'train' and np.random.rand(1)[0] < self.aug:
            clouds, action_tcps, pose, past_action_tcps, past_pose = self._augmentation(clouds, action_tcps, pose, past_action_tcps, past_pose)
        if vis_pcd_flag:
            pcd_vis, index_vis = fps(clouds[-1][:,:3], 3000)
            index_vis = index_vis.detach().cpu().numpy()[0]
            visualize(vis, pcd_vis, win=2,  opts={'markersize': 2, 'markercolor': color_vis}) 

        # rotation transformation (to 6d)
        action_tcps = xyz_rot_transform(action_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
        actions = np.concatenate((action_tcps, action_grippers), axis = -1)          # (L=20, dim=10)

        if past_action_tcps is not None:
            past_action_tcps = xyz_rot_transform(past_action_tcps, from_rep = "quaternion", to_rep = "rotation_6d")                       
            past_actions = np.concatenate((past_action_tcps, past_action_grippers), axis = -1)  # (L=20, dim=10)
        else:
            past_actions_normalized = np.zeros(shape=(cond_seq_len, self.action_dim), dtype=np.float32)
            past_pose_normalized = np.zeros(shape=(cond_seq_len, self.pose_dim), dtype=np.float32)

        # normalization
        actions_normalized = self._normalize_tcp(actions.copy())
        norm_pose_func = self._normalize_pose_v2 if 'pour_balls' in self.task_name else self._normalize_pose
        pose_normalized = norm_pose_func(pose.copy())   # (cond_seq_len, num_target, 4, 4) --> (cond_seq_len, num_target*pose_dim)
        
        if past_action_tcps is not None:
            past_actions_normalized = self._normalize_tcp(past_actions.copy())
            past_pose_normalized = norm_pose_func(past_pose.copy())

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
            ret_dict[key] = collate_fn([d[key] for d in batch]) # replace the above 4 lines

        coords_batch = ret_dict['input_coords_list']            # list of array, list_length = batch * num_obs
        feats_batch = ret_dict['input_feats_list']
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)  # (total_num_pts, 4), (total_num_pts, 6)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def decode_gripper_width(gripper_width):
    return gripper_width / 1000. * 0.095


if __name__ == '__main__':
    task_name = 'hang_mug_sampled'  # 'pour_balls_sampled'  # 'arrange_truck_sampled'  # 
    dataset = RealDataset(task_name, num_obs=1, horizon=16, aug=True)
    for i in range(len(dataset)):
        sample = dataset[i]
    print('test done')
