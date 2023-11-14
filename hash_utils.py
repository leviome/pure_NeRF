import math
import pdb
from math import exp, log, floor

import torch
from torch.optim.optimizer import Optimizer


def trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
    '''
    x: B x 3
    voxel_min_vertex: B x 3
    voxel_max_vertex: B x 3
    voxel_embedds: B x 8 x 2
    '''
    # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
    weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 3

    # step 1
    # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
    c00 = voxel_embedds[:, 0] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 4] * weights[:, 0][:, None]
    c01 = voxel_embedds[:, 1] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 5] * weights[:, 0][:, None]
    c10 = voxel_embedds[:, 2] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 6] * weights[:, 0][:, None]
    c11 = voxel_embedds[:, 3] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 7] * weights[:, 0][:, None]

    # step 2
    c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
    c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

    # step 3
    c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

    return c


def make_hash(coords, log2_hashmap_size):
    """
    Args:
        coords: this function can process upto 7 dim coordinates
        log2_hashmap_size: logarithm of T w.r.t 2

    Returns:
    """
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    """
    Args:
        xyz: 3D coordinates of samples. B x 3
        bounding_box: min and max x,y,z coordinates of object bbox
        resolution: number of voxels per axis
        log2_hashmap_size:

    Returns:
    """
    box_min, box_max = bounding_box

    keep_mask = xyz == torch.max(torch.min(xyz, box_max), box_min)  # is_in_box
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max - box_min) / resolution.to(bounding_box)
    box_offset = torch.tensor([[[0, 0, 0], [0, 0, 1],
                                [0, 1, 0], [0, 1, 1],
                                [1, 0, 0], [1, 0, 1],
                                [1, 1, 0], [1, 1, 1]]])

    bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
    voxel_min_vertex = bottom_left_idx * grid_size + box_min
    addition = torch.tensor([1.0, 1.0, 1.0])
    addition = addition.to(voxel_min_vertex)
    voxel_max_vertex = voxel_min_vertex + addition * grid_size
    box_offset = box_offset.to(bottom_left_idx.device)
    voxel_indices = bottom_left_idx.unsqueeze(1) + box_offset
    hashed_voxel_indices = make_hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


def total_variation_loss(embeddings, device, min_resolution, max_resolution, level, log2_hashmap_size, n_levels=16):
    # Get resolution
    b = exp((log(max_resolution) - log(min_resolution)) / (n_levels - 1))
    resolution = torch.tensor(floor(min_resolution * b ** level)).to(device)

    # Cube size to apply TV loss
    min_cube_size = min_resolution - 1
    max_cube_size = 50  # can be tuned
    if min_cube_size > max_cube_size:
        print("ALERT! min cuboid size greater than max!")
        pdb.set_trace()
    cube_size = torch.floor(torch.clip(resolution / 10.0, min_cube_size, max_cube_size)).int().to(device)

    # Sample cuboid
    min_vertex = torch.randint(0, resolution - cube_size, (3,))
    idx = min_vertex + torch.stack([torch.arange(cube_size + 1) for _ in range(3)], dim=-1)
    cube_indices = torch.stack(torch.meshgrid(idx[:, 0], idx[:, 1], idx[:, 2]), dim=-1).to(device)

    hashed_indices = make_hash(cube_indices, log2_hashmap_size)
    cube_embeddings = embeddings(hashed_indices)
    # hashed_idx_offset_x = hash(idx+torch.tensor([1,0,0]), log2_has hmap_size)
    # hashed_idx_offset_y = hash(idx+torch.tensor([0,1,0]), log2_hashmap_size)
    # hashed_idx_offset_z = hash(idx+torch.tensor([0,0,1]), log2_hashmap_size)

    # Compute loss
    # tv_x = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_x), 2).sum()
    # tv_y = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_y), 2).sum()
    # tv_z = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_z), 2).sum()
    tv_x = torch.pow(cube_embeddings[1:, :, :, :] - cube_embeddings[:-1, :, :, :], 2).sum()
    tv_y = torch.pow(cube_embeddings[:, 1:, :, :] - cube_embeddings[:, :-1, :, :], 2).sum()
    tv_z = torch.pow(cube_embeddings[:, :, 1:, :] - cube_embeddings[:, :, :-1, :], 2).sum()

    return (tv_x + tv_y + tv_z) / cube_size
