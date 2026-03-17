import torch
import torch.nn as nn

from policy.tokenizer import Sparse3DEncoder
from policy.transformer import Transformer, Transformer_X
from policy.diffusion import DiffusionUNetPolicy
from dataset.dataset_utils import LinearNormalizer
from typing import List, Type
from termcolor import cprint


def print_params(model):
    """
    Copy from dp3. Print the number of parameters in each part of the model.
    """
    params_dict = {}
    all_num_param = sum(p.numel() for p in model.parameters())

    for name, param in model.named_parameters():
        part_name = name.split(".")[0]
        if part_name not in params_dict:
            params_dict[part_name] = 0
        params_dict[part_name] += param.numel()

    cprint(f"----------------------------------", "cyan")
    cprint(f"Class name: {model.__class__.__name__}", "cyan")
    cprint(f"  Number of parameters: {all_num_param / 1e6:.4f}M", "cyan")
    for part_name, num_params in params_dict.items():
        cprint(
            f"   {part_name}: {num_params / 1e6:.4f}M ({num_params / all_num_param:.2%})",
            "cyan",
        )
    cprint(f"----------------------------------", "cyan")


class RISE(nn.Module):
    def __init__(
        self, 
        horizon = 16,
        num_action = 8,
        input_dim = 6,
        obs_feature_dim = 512, 
        action_dim = 10, 
        hidden_dim = 512,   # unused
        nheads = 8, 
        num_encoder_layers = 4, 
        num_decoder_layers = 1, 
        dim_feedforward = 2048, 
        dropout = 0.1,
        num_obs = 1,
    ):
        super().__init__()
        assert num_obs == 1  
        hidden_dim = obs_feature_dim
        self.num_action = num_action
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim, num_obs)                                           # 14.6M   
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout) # 16.8M   
        self.action_decoder = DiffusionUNetPolicy(action_dim, horizon, num_obs, obs_feature_dim)                             # 19.5M
        self.readout_embed = nn.Embedding(1, hidden_dim)
        self.normalizer = LinearNormalizer()
        self.have_normalizer = False
        print_params(self)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.have_normalizer = True

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def forward(self, cloud, actions = None, batch_size = 24):
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)          # (batch, max_token, dim)，mask: (batch, max_token)
        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]   # (1, B, L=1, D) --> (B, L=1, D)
        readout = readout[:, 0]     # (B, L=1,  D=512) --> (B, D)
        if actions is not None:     # (B, L=20, D=10)
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            # return action_pred
            return action_pred[:, :self.num_action]
        

def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """ 
    Copy from dp3. Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class XRISE(nn.Module):
    def __init__(
        self, 
        horizon = 16,
        num_action = 8,
        num_pose = 8,
        pose_dim = 9,
        aug_flag = True,
        action_aug_std = 0.05,
        pose_t_aug_std = 0.05,
        pose_R_aug_std = 0.1,
        input_dim = 6,
        obs_feature_dim = 512, 
        action_dim = 10, 
        hidden_dim = 512, # unused
        nheads = 8, 
        num_encoder_layers = 4, 
        num_decoder_layers = 1, 
        dim_feedforward = 2048, 
        dropout = 0.1,
        num_obs = 1,
    ):
        super().__init__()
        assert num_obs == 1
        hidden_dim = obs_feature_dim
        self.num_action = num_action
        self.num_pose = num_pose
        self.aug_flag = aug_flag
        self.action_aug_std = action_aug_std
        self.pose_R_aug_std = pose_R_aug_std                                                                              
        self.pose_t_aug_std = pose_t_aug_std
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim, num_obs)                                                         # 14.6M
        self.transformer   = Transformer_X(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout) # 16.8M   
        self.action_decoder = DiffusionUNetPolicy(action_dim, horizon, num_obs, obs_feature_dim)                                           # 19.5M
        self.pose_decoder = DiffusionUNetPolicy(pose_dim, horizon, num_obs, obs_feature_dim)                             
        self.readout_embed = nn.Embedding(1, hidden_dim)
        
        self.normalizer = LinearNormalizer()
        self.have_normalizer = False

        pose_mlp_in  = num_pose * pose_dim
        pose_mlp_out = obs_feature_dim
        action_mlp_in  = num_action * action_dim
        action_mlp_out = obs_feature_dim
        self.pose_mlp   = nn.Sequential(*create_mlp(pose_mlp_in, pose_mlp_out, [256]))
        self.action_mlp = nn.Sequential(*create_mlp(action_mlp_in, action_mlp_out, [256]))

        print_params(self)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.have_normalizer = True

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def forward(self, observations, actions = None, poses = None, batch_size = 24):
        # prepare data
        cloud = observations['cloud']
        pose_have_past   = observations['pose_have_past'].squeeze(1)   # (B, 1) --> (B)  
        action_have_past = observations['action_have_past'].squeeze(1) # (B, 1) --> (B)  

        if torch.all(observations['pose_past_normalized'] == 0):       # start of inference
            npose_past = None
        else:                                                          # training or inference 
            npose_past = observations['pose_past_normalized']          # (B, L, D=9|18)
        
        if torch.all(observations['action_past_normalized'] == 0):
            naction_past = None
        else:
            naction_past = observations['action_past_normalized']      # (B, L, D=14)
        
        # augment conditional sequence when training
        if actions is not None and self.aug_flag and self.action_aug_std > 0:
            assert naction_past is not None
            naction_past = naction_past + torch.randn_like(naction_past) * self.action_aug_std
        
        if actions is not None and self.aug_flag and self.pose_R_aug_std > 0:
            assert npose_past is not None
            assert self.pose_t_aug_std > 0
            if npose_past.shape[-1] % 9 == 0:
                npose_noise_R = torch.randn_like(npose_past)[...,:6] * self.pose_R_aug_std
                npose_noise_t = torch.randn_like(npose_past)[...,-3:] * self.pose_t_aug_std
                num_pose = npose_past.shape[-1] // 9
                npose_past = npose_past + torch.cat([npose_noise_R, npose_noise_t] * num_pose, dim=2)
            elif npose_past.shape[-1] == 6:     # for real-world task: pour balls
                npose_noise_R = torch.randn_like(npose_past)[...,:3] * self.pose_R_aug_std
                npose_noise_t = torch.randn_like(npose_past)[...,-3:] * self.pose_t_aug_std
                num_pose = 1
                npose_past = npose_past + torch.cat([npose_noise_R, npose_noise_t] * num_pose, dim=2)

        # extract point cloud features
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)              # (batch, max_token, dim)，mask: (batch, max_token)
        readout, memory = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)   # (1, B, L=1, D) --> (B, L=1, D)
        readout = readout[-1][:, 0]     # (B, 1, D=512) --> (B, D)
        
        # feature fusion
        if npose_past is not None:
            readout_embed_p = self.pose_mlp(npose_past.reshape(batch_size, -1))
            cond_for_action, _ = self.transformer(src, src_padding_mask, readout_embed_p.unsqueeze(1), pos, memory)
            cond_for_action = cond_for_action[-1][:, 0]
            if (~pose_have_past).any():
                cond_for_action[~pose_have_past] = readout[~pose_have_past]
        else:   
            cond_for_action = readout

        if naction_past is not None:
            readout_embed_a = self.action_mlp(naction_past.reshape(batch_size, -1))
            cond_for_pose, _ = self.transformer(src, src_padding_mask, readout_embed_a.unsqueeze(1), pos, memory)
            cond_for_pose = cond_for_pose[-1][:, 0]
            if (~action_have_past).any():
                cond_for_pose[~action_have_past] = readout[~action_have_past]
        else:   
            cond_for_pose = readout
        
        # compute loss or perform inference
        if actions is not None:     # (B, L, D)
            assert poses is not None
            loss_action = self.action_decoder.compute_loss(cond_for_action, actions)
            loss_pose = self.pose_decoder.compute_loss(cond_for_pose, poses)
            return loss_action + loss_pose
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(cond_for_action)
                pose_pred = self.pose_decoder.predict_action(cond_for_pose)
            return action_pred[:, :self.num_action], pose_pred[:, :self.num_pose]  # for robotwin evaluation: num_execute_steps = num_action = length of conditional sequence
            # return action_pred, pose_pred   # for real-world evaluation

