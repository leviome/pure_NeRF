import os

import numpy as np
import torch
from tqdm import tqdm

from camera_trans import CamTrans, pose_spherical_v2
from dataset import DataForNeRF
from nerf import NeRFWrapper


def simple_config():
    configs = dict()
    configs["basedir"] = "../data/nerf_synthetic/lego/"
    configs["near"] = 2.0
    configs["far"] = 6.0
    configs["render_a_view"] = True
    return configs


def _main(render_only=False):
    cfg = simple_config()
    device = "cuda:2"

    print("loading data...")
    dataset = DataForNeRF(root=cfg["basedir"])

    w, h = dataset.w, dataset.h
    focal = dataset.focal
    cam_trans = CamTrans(focal=focal, w=w, h=h, device=device)

    i_train, i_val, i_test = dataset.i_split
    model = NeRFWrapper(device)
    global_step = model.start

    model.load_checkpoint("./checkpoints/200000.tar")
    if render_only:
        if cfg["render_a_view"]:
            pose = dataset.poses[0, :3, :4]
            model.render_a_view(cam_trans, pose)
        else:
            poses = torch.stack([pose_spherical_v2(angle, -30.0, 4.0)
                                 for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
            model.render_a_path(cam_trans, poses)
        return

    print('Begin training...')
    # print('TRAIN views are', i_train)
    # print('TEST views are', i_test)
    # print('VAL views are', i_val)
    for i in range(model.start, 300000 + 1):
        img_i = np.random.choice(i_train)
        target = torch.Tensor(dataset.images[img_i]).to(device)
        pose = dataset.poses[img_i, :3, :4]

        rays_o, rays_d = cam_trans.get_rays(pose)
        batch_o, batch_d, target_s = cam_trans.random_rays(rays_o, rays_d, target,
                                                           n_rand=1024 * 16)

        viewdirs = batch_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        batch_o = torch.reshape(batch_o, [-1, 3]).float()
        batch_d = torch.reshape(batch_d, [-1, 3]).float()

        near, far = cfg["near"] * torch.ones_like(batch_d[..., :1]), cfg["far"] * torch.ones_like(batch_d[..., :1])
        rays = torch.cat([batch_o, batch_d, near, far, viewdirs], -1)
        rgb_map = model.render_rays(rays)
        model.optimizer.zero_grad()
        img_loss = torch.mean((rgb_map - target_s) ** 2)  # MSE

        img_loss.backward()
        model.optimizer.step()
        psnr = -10. * torch.log(img_loss) / torch.log(torch.Tensor([10.]).to(img_loss))

        decay_rate = 0.1
        lrate_decay = 250
        decay_steps = lrate_decay * 1000
        lrate = 5e-4
        new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = new_lrate

        if i % 1000 == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {img_loss.item()}  PSNR: {psnr.item()}")

        if i % 10000 == 0:
            path = os.path.join("checkpoints", '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': model.net.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    _main()
