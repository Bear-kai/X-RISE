import numpy as np
import h5py
import os
# import numba
from typing import Dict, Callable, Union
import torch
import torch.nn as nn
import zarr

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from parse_hdf5 import read_hdf5


def dict_apply(x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
    """ usage: dict_apply(data, torch.from_numpy) """
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def load_hdf5(dataset_path):    # joint action
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (                      # eg.(N=118,); (N=118, 6)
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (                    # eg.(N=118,); (N=118, 6)
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        vector = root["/joint_action/vector"][()]       # eg.(N=118, dim=14);  dual arms: (6+1)x2=14
        pointcloud = root["/pointcloud"][()]            # eg.(N=118, 1024, 6)

        # obtain object pose
        object_pose_ls = []
        object_pose1 = root["/observation/target_obj1_pose"][()]      # eg.(N=118, 4, 4)
        object_pose_ls.append(object_pose1)
        if "target_obj2_pose" in root["observation"].keys():
            object_pose2 = root["/observation/target_obj2_pose"][()]  # eg.(N=118, 4, 4)
            object_pose_ls.append(object_pose2)
        if "target_obj3_pose" in root["observation"].keys():
            object_pose3 = root["/observation/target_obj3_pose"][()]  # eg.(N=118, 4, 4)
            object_pose_ls.append(object_pose3)

        # Find which robotic arm to use for the task: 1 ~ full opening, 0 ~ full closing.
        use_left_arm = (left_gripper < 1).any()
        use_right_arm = (right_gripper < 1).any()
        if use_left_arm and use_right_arm:
            use_arm = 3 # 'both'
        elif use_left_arm:
            use_arm = 1 # 'left'
        elif use_right_arm:
            use_arm = 2 # 'right'
        else:
            raise ValueError("No arm is used.")

    # return left_gripper, left_arm, right_gripper, right_arm, vector, pointcloud, object_pose_ls
    return vector, pointcloud, object_pose_ls, use_arm


def load_hdf5_v2(dataset_path): # tcp action
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    data = read_hdf5(dataset_path)

    left_endpose = data["endpose"]["left_endpose"].astype(np.float32)       # eg.(N=118, dim=7)，3d_trans + 4d_quat
    left_gripper = data["endpose"]["left_gripper"].astype(np.float32)       # eg.(N=118,)
    right_endpose = data["endpose"]["right_endpose"].astype(np.float32)     # eg.(N=118, dim=7)
    right_gripper = data["endpose"]["right_gripper"].astype(np.float32)     # eg.(N=118,)
    vector = np.concatenate([left_endpose, right_endpose, left_gripper[:,None], right_gripper[:,None]], axis=-1) # eg.(N=118, dim=16)

    # Find which robotic arm to use for the task: 1 ~ full opening, 0 ~ full closing.
    use_left_arm = (left_gripper < 1).any()
    use_right_arm = (right_gripper < 1).any()
    if use_left_arm and use_right_arm:
        use_arm = 3 # 'both'
    elif use_left_arm:
        use_arm = 1 # 'left'
    elif use_right_arm:
        use_arm = 2 # 'right'
    else:
        use_arm = 'none'
    # print(f"Use arm: {use_arm}")

    pointcloud = data["pointcloud"]                      # eg.(N=118, 1024, 6)
    
    # vis
    if False:
        head_cam  = data["observation"]["head_camera"]
        front_cam = data["observation"]["front_camera"]
        import PIL.Image as Image
        frame0 = head_cam['rgb'][0]
        Image.fromarray(frame0).save('frame0.png')
        video_path = '/root/RoboTwin/data/beat_block_hammer/demo_clean/video/episode0.mp4'
        import cv2
        cap = cv2.VideoCapture(video_path)
        num = 0
        while cap.isOpened():
            ret, frame = cap.read() # bgr format
            num += 1
            if not ret:
                break
            Image.fromarray(frame[...,::-1]).save(f'frame_from_video.png')
        cap.release()

    # obtain object pose
    object_pose_ls = []
    object_pose1 = data["observation"]["target_obj1_pose"]        # eg.(N=118, 4, 4)
    object_pose_ls.append(object_pose1)
    if "target_obj2_pose" in data["observation"].keys():
        object_pose2 = data["observation"]["target_obj2_pose"]    # eg.(N=118, 4, 4)
        object_pose_ls.append(object_pose2)
    if "target_obj3_pose" in data["observation"].keys():
        object_pose3 = data["observation"]["target_obj3_pose"]    # eg.(N=118, 4, 4)
        object_pose_ls.append(object_pose3)

    return vector, pointcloud, object_pose_ls, use_arm


def get_flat_pose(target_pose):
    # target_pose: (N, 4, 4)
    pose = np.concatenate([target_pose[:,:3,0], target_pose[:,:3,1], target_pose[:,:3,-1]], axis=-1)    # (N, 9)
    return pose

def get_flat_pose_v2(target_pose, axis=1):
    # target_pose: (N, 4, 4)
    # axis: which axis to choose
    assert axis in [0,1,2]  # corresponds to xyz axis
    pose = np.concatenate([target_pose[:,:3, axis], target_pose[:,:3,-1]], axis=-1)    # (N, 6)
    return pose


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


# @numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,
    use_arm_ls: list,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
    debug: bool = True,
) -> np.ndarray:
    assert episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)     

    indices = list()
    use_arm_arr = list()

    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after - 1   # subtract 1 here

        use_arm = use_arm_ls[i]
        
        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx,
                start_idx,
                end_idx,
            ])     
            use_arm_arr.append(use_arm)
    indices = np.array(indices)
    use_arm_arr = np.array(use_arm_arr).reshape(-1,1)  # shape:(N,1)
    return indices, use_arm_arr


# -----------------------


class DictOfTensorMixin(nn.Module):

    def __init__(self, params_dict=None):
        super().__init__()
        if params_dict is None:
            params_dict = nn.ParameterDict()
        self.params_dict = params_dict

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _load_from_state_dict(  # override nn.Module._load_from_state_dict
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return

            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)

        def load_dict(state_dict, prefix):
            out_dict = nn.ParameterDict()
            for key, value in state_dict.items():
                value: torch.Tensor
                if key.startswith(prefix):
                    param_keys = key[len(prefix):].split(".")[1:]
                    # if len(param_keys) == 0:
                    #     import pdb; pdb.set_trace()
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict

        self.params_dict = load_dict(state_dict, prefix + "params_dict")
        self.params_dict.requires_grad_(False)
        return


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] = _fit(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                )
        else:
            self.params_dict["_default"] = _fit(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
            )

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: "SingleFieldLinearNormalizer"):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if "_default" not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict["_default"]
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]

        result = dict()
        for key, value in self.params_dict.items():
            if key != "_default":
                result[key] = value["input_stats"]
        return result

    def get_output_stats(self, key="_default"):
        input_stats = self.get_input_stats()
        if "min" in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)

        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key: value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        self.params_dict = _fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )

    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj

    @classmethod
    def create_manual(
        cls,
        scale: Union[torch.Tensor, np.ndarray],
        offset: Union[torch.Tensor, np.ndarray],
        input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    ):

        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x

        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype

        params_dict = nn.ParameterDict({
            "scale": to_tensor(scale),
            "offset": to_tensor(offset),
            "input_stats": nn.ParameterDict(dict_apply(input_stats_dict, to_tensor)),
        })
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict["input_stats"]

    def get_output_stats(self):
        return dict_apply(self.params_dict["input_stats"], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)


def _fit(
    data: Union[torch.Tensor, np.ndarray, zarr.Array],
    last_n_dims=1,
    dtype=torch.float32,
    mode="limits",
    output_max=1.0,
    output_min=-1.0,
    range_eps=1e-4,
    fit_offset=True,
):
    assert mode in ["limits", "gaussian"]
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == "limits":
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == "gaussian":
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = -input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    # save
    this_params = nn.ParameterDict({
        "scale":
        scale,
        "offset":
        offset,
        "input_stats":
        nn.ParameterDict({
            "min": input_min,
            "max": input_max,
            "mean": input_mean,
            "std": input_std,
        }),
    })
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    assert "scale" in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


def test():
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode="limits", last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.0)
    assert np.allclose(datan.min(), -1.0)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode="limits", last_n_dims=1, fit_offset=False)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.0, atol=1e-3)
    assert np.allclose(datan.min(), 0.0, atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    data = torch.zeros((100, 10, 9, 2)).uniform_()
    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode="gaussian", last_n_dims=0)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.mean(), 0.0, atol=1e-3)
    assert np.allclose(datan.std(), 1.0, atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    # dict
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    normalizer = LinearNormalizer()
    normalizer.fit(data, mode="limits", last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.0)
    assert np.allclose(datan.min(), -1.0)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    data = {
        "obs": torch.zeros((1000, 128, 9, 2)).uniform_() * 512,
        "action": torch.zeros((1000, 128, 2)).uniform_() * 512,
    }
    normalizer = LinearNormalizer()
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    dataun = normalizer.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    state_dict = normalizer.state_dict()
    n = LinearNormalizer()
    n.load_state_dict(state_dict)
    datan = n.normalize(data)
    dataun = n.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)
