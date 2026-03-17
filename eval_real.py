import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME

from termcolor import cprint
from copy import deepcopy
from easydict import EasyDict as edict

from policy.policy import RISE
from eval_agent import Agent
from utils.constants import *
from utils.training import set_seed
from dataset.projector import Projector
from utils.ensemble import EnsembleBuffer
from utils.transformation import rotation_transform
from dataset.constants import INTRINSICS


default_args = edict({
    "task_name": "hang_mug_sampled", # "pour_balls_sampled" # "arrange_truck_sampled" # 
    "ckpt": "path/to/hang_mug_sampled-demo_clean-50_0/policy_last.ckpt",
    "calib": "path/to/calib/1770283618030",  # 1752812316696 # 1770283618030
    
    "horizon": 16,            # the number of prediction steps
    "num_inference_step": 16, # the number of execution  steps, keep the same with num_action  
    "num_action": 16,         # the number of execution  steps
    "action_dim": 10,

    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "max_steps": 300,
    "seed": 233,
    "vis": True, # False,
    "discretize_rotation": True,
    "ensemble_mode": "new",
})


def create_point_cloud(colors, depths, cam_intrinsics, voxel_size = 0.005):
    """
    color, depth => point cloud
    """
    h, w = depths.shape
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale = 1.0, convert_rgb_to_intensity = False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points).astype(np.float32)
    colors = np.array(cloud.colors).astype(np.float32)

    x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
    y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
    z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
    mask = (x_mask & y_mask & z_mask)
    points = points[mask]
    colors = colors[mask]
    # imagenet normalization
    colors = (colors - IMG_MEAN) / IMG_STD
    # final cloud
    # cloud_final = np.concatenate([points, colors], axis = -1).astype(np.float32)
    return points, colors # cloud_final

def create_batch(coords, feats):
    """
    coords, feats => batch coords, batch feats (batch size = 1)
    """
    coords_batch = [coords]
    feats_batch = [feats]
    coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
    return coords_batch, feats_batch

def create_input(colors, depths, cam_intrinsics, projector, cam_id, voxel_size = 0.005):
    """
    colors, depths => batch coords, batch feats
    """
    points, colors = create_point_cloud(colors, depths, cam_intrinsics, voxel_size = voxel_size)
    
    # crop point cloud: 1) transform to base frame; 2) manually crop the desktop points; 3) back to camera frame.
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
    points = points[points_base[2] >= 0.005]
    colors = colors[points_base[2] >= 0.005]
    cloud = np.concatenate([points, colors], axis=-1).astype(np.float32)

    coords = np.ascontiguousarray(cloud[:, :3] / voxel_size, dtype = np.int32)
    coords_batch, feats_batch = create_batch(coords, cloud[:,:3])   # exclude rgb

    return coords_batch, feats_batch, cloud

def unnormalize_action(action):
    action[..., :3] = (action[..., :3] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    action[..., -1] = (action[..., -1] + 1) / 2.0 * MAX_GRIPPER_WIDTH
    return action

def rot_diff(rot1, rot2):
    rot1_mat = rotation_transform(
        rot1,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    rot2_mat = rotation_transform(
        rot2,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    diff = rot1_mat @ rot2_mat.T
    diff = np.diag(diff).sum()
    diff = min(max((diff - 1) / 2.0, -1), 1)
    return np.arccos(diff)

def discretize_rotation(rot_begin, rot_end, rot_step_size = np.pi / 16):
    n_step = int(rot_diff(rot_begin, rot_end) // rot_step_size) + 1
    rot_steps = []
    for i in range(n_step):
        rot_i = rot_begin * (n_step - 1 - i) / n_step + rot_end * (i + 1) / n_step
        rot_steps.append(rot_i)
    return rot_steps

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
        num_obs = 1,
    ).to(device)
    
    assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
    policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
    cprint("Checkpoint {} loaded.".format(args.ckpt), "magenta")

    # if args.data_style == 'dp3':
    #     policy.have_normalizer = True
    # policy.eval()

    return policy

def evaluate():
    args = deepcopy(default_args)
    
    cam_id = "104122063550"
    args.pose_dim = 6 if 'pour_balls' in args.task_name else 9   # hard code

    # set up device
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load policy
    print("Loading policy ...")
    policy = get_policy_model(args, device)

    # evaluation
    agent = Agent(
        robot_serial="Rizon4-062027",
        gripper_port = "/dev/ttyUSB0",
        camera_serial = cam_id # "750612070851"
    )

    projector = Projector(args.calib)
    ensemble_buffer = EnsembleBuffer(mode = args.ensemble_mode)
    
    if args.discretize_rotation:
        last_rot = np.array(agent.ready_rot_6d, dtype = np.float32)

    with torch.inference_mode():
        policy.eval()
        prev_width = None
        for t in range(args.max_steps):
            if t % args.num_inference_step == 0:
                # pre-process inputs
                colors, depths = agent.get_observation()
                coords, feats, cloud = create_input(colors, depths, INTRINSICS[cam_id], projector, cam_id, args.voxel_size) # cloud用于open3d可视化
                feats, coords = feats.to(device), coords.to(device)
                cloud_data = ME.SparseTensor(feats, coords)

                # predict
                pred_raw_action = policy(cloud_data, actions = None, batch_size = 1).squeeze(0).cpu().numpy()

                # unnormalize predicted actions
                action = unnormalize_action(pred_raw_action)

                # visualization
                if args.vis:
                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] * IMG_STD + IMG_MEAN)
                    tcp_vis_list = []
                    for raw_tcp in action:
                        tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[:3])
                        tcp_vis_list.append(tcp_vis)
                    o3d.visualization.draw_geometries([pcd, *tcp_vis_list])

                # project action to base coordinate
                action_tcp = projector.project_tcp_to_base_coord(action[..., :-1], cam=cam_id, rotation_rep="rotation_6d")
                action_width = action[..., -1]
                # safety insurance
                action_tcp[..., :3] = np.clip(action_tcp[..., :3], SAFE_WORKSPACE_MIN + SAFE_EPS, SAFE_WORKSPACE_MAX - SAFE_EPS)
                # full actions
                action = np.concatenate([action_tcp, action_width[..., np.newaxis]], axis = -1)
                # add to ensemble buffer
                ensemble_buffer.add_action(action, t)
            
            # get step action from ensemble buffer
            step_action = ensemble_buffer.get_action()
            if step_action is None:   # no action in the buffer => no movement.
                continue
            
            step_tcp = step_action[:-1]
            step_width = step_action[-1]

            # print debug:
            # step_quat = rotation_transform(step_tcp[3:], from_rep = "rotation_6d", to_rep = "quaternion")
            # print(f'tcp_translation: {step_tcp[:3]}; tcp_quat: {step_quat}; gripper_width: {step_width:.3f}')

            # send tcp pose to robot
            if args.discretize_rotation:
                rot_steps = discretize_rotation(last_rot, step_tcp[3:], np.pi / 16)
                last_rot = step_tcp[3:]
                for rot in rot_steps:
                    step_tcp[3:] = rot
                    agent.set_tcp_pose(
                        step_tcp, 
                        rotation_rep = "rotation_6d",
                        blocking = True
                    )
            else:
                agent.set_tcp_pose(
                    step_tcp,
                    rotation_rep = "rotation_6d",
                    blocking = True
                )

            # if step_width < 0.01:
            #     print(f'manually close gripper: set step_width={step_width:.3f} to zero.')
            #     step_width = 0
            
            # send gripper width to gripper (thresholding to avoid repeating sending signals to gripper)
            if prev_width is None or abs(prev_width - step_width) > GRIPPER_THRESHOLD:
                print('send_gripper...')
                agent.set_gripper_width(step_width, blocking = True)
                prev_width = step_width
    
    agent.stop()



if __name__ == '__main__':
    evaluate()