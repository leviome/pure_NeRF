import os
import os.path as osp

import numpy as np
import torch
from tqdm import tqdm, trange
from time import time

from camera_trans import CamTrans, pose_spherical_v2
from dataset import DataForNeRF
from nerf import NeRFWrapper
from hash_utils import total_variation_loss

from torch.utils.tensorboard import SummaryWriter
import datetime 
import argparse


log_writer = SummaryWriter()

def get_args():
    parser = argparse.ArgumentParser('demo', add_help=False)
    parser.add_argument('--basedir', default="./data/nerf_synthetic/lego/", type=str)
    parser.add_argument('--near', default=2.0, type=float)
    parser.add_argument('--far', default=6.0, type=float)
    parser.add_argument('--hash', action="store_true")
    parser.add_argument('--render_only', action="store_true")
    parser.add_argument('--render_a_view', action="store_true")
    parser.add_argument('--load_ckpt', action="store_true")
    parser.add_argument('--lrate', default=0.01, type=float)
    parser.add_argument('--lrate_decay', default=10, type=float)
    parser.add_argument('--tv_loss_weight', default=1e-6, type=float)

    parser.add_argument('--device', default="0", type=str)

    return parser.parse_args()


def simple_config():
    configs = dict()
    configs["basedir"] = "./data/nerf_synthetic/lego/"
    configs["near"] = 2.0
    configs["far"] = 6.0
    configs["use_hash"] = True

    configs["render_only"] = False
    configs["render_a_view"] = True

    # lrate can be 0.01 for hash mode
    configs["lrate"] = 0.01
    configs["lrate_decay"] = 10

    configs["tv_loss_weight"] = 1e-06
    return configs


def _main():
    args = get_args()
    print(args)

    dt = datetime.datetime.now().date()
    is_hash = "hash" if args.hash else "vanilla"
    save_path = osp.join("checkpoints", str(dt) + '-' + is_hash)
    if not osp.exists(save_path):
        os.makedirs(save_path)

    device = "cuda:" + args.device

    model = NeRFWrapper(device, use_hash=args.hash, initial_lr=args.lrate,
                        near=args.near, far=args.far)

    print("loading data...")
    dataset = DataForNeRF(root=args.basedir, half_res=True)

    w, h = dataset.w, dataset.h
    focal = dataset.focal
    cam_trans = CamTrans(focal=focal, w=w, h=h, device=device)

    i_train, i_val, i_test = dataset.i_split

    if args.load_ckpt:
        model.load_checkpoint(osp.join(save_path, "008000.tar"))
    global_step = model.start

    if args.render_only:
        if args.render_a_view:
            pose = dataset.poses[0, :3, :4]
            model.render_a_view(cam_trans, pose, save_name="demo_hash1.png",
                                h=h, w=w)
        else:
            poses = torch.stack([pose_spherical_v2(angle, -30.0, 4.0)
                                 for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
            model.render_a_path(cam_trans, poses)
        return

    print('Begin training...')
    # print('TRAIN views are', i_train)
    # print('TEST views are', i_test)
    # print('VAL views are', i_val)
    tic = time()
    for i in trange(model.start, 200000 + 1):
        img_i = np.random.choice(i_train)
        target = torch.Tensor(dataset.images[img_i]).to(device)
        pose = dataset.poses[img_i, :3, :4]

        rays_o, rays_d = cam_trans.get_rays(pose)
        batch_o, batch_d, target_s = cam_trans.random_rays(rays_o, rays_d, target,
                                                           n_rand=1024)

        viewdirs = batch_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        batch_o = torch.reshape(batch_o, [-1, 3]).float()
        batch_d = torch.reshape(batch_d, [-1, 3]).float()

        near, far = args.near * torch.ones_like(batch_d[..., :1]), args.far * torch.ones_like(batch_d[..., :1])
        rays = torch.cat([batch_o, batch_d, near, far, viewdirs], -1)
        rgb_map, rgb_fine, _ = model.render_rays(rays)
        model.optimizer.zero_grad()
        img_loss = torch.mean((rgb_map - target_s) ** 2)  # MSE
        img_fine_loss = torch.mean((rgb_fine - target_s) ** 2)  # MSE
        loss = img_loss + img_fine_loss

        if args.hash:
            n_levels = model.embedder.n_levels
            min_res = model.embedder.base_resolution
            max_res = model.embedder.finest_resolution
            log2_hashmap_size = model.embedder.log2_hashmap_size
            TV_loss = sum(total_variation_loss(model.embedder.embeddings[i],
                                               model.device,
                                               min_res, max_res,
                                               i, log2_hashmap_size,
                                               n_levels=n_levels) for i in range(n_levels))
            loss = loss + args.tv_loss_weight * TV_loss
            if i > 1000:
                args.tv_loss_weight = 0.0

        loss.backward()
        model.optimizer.step()

        psnr = -10. * torch.log(img_fine_loss) / torch.log(torch.Tensor([10.]).to(img_fine_loss))

        decay_rate = 0.1
        lrate_decay = args.lrate_decay
        decay_steps = lrate_decay * 1000
        lrate = args.lrate
        new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
        global_step += 1

        for param_group in model.optimizer.param_groups:
            param_group['lr'] = new_lrate

        if i % 100 == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {img_fine_loss.item()}  PSNR: {psnr.item()}")
            log_writer.add_scalar('Loss/train', float(loss), i)
            log_writer.add_scalar('PSNR/train', float(psnr.item()), i)

        time_consumption = int(time() - tic)
        if time_consumption > 0 and time_consumption % 60 == 0:
            log_writer.add_scalar('PSNR_time/train', float(psnr.item()), time_consumption // 60)

        if i % 1000 == 0 and i > 0:
            path = osp.join(save_path, '{:06d}.tar'.format(i))
            if args.hash:
                torch.save({
                    'global_step': global_step,
                    'hash_em': model.embedder.state_dict(),
                    'net_state_dict': model.net.state_dict(),
                    'net_fine_state_dict': model.net_fine.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'net_state_dict': model.net.state_dict(),
                    'net_fine_state_dict': model.net_fine.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    _main()
