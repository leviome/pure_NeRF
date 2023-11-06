import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def volume_rendering(raw, z_vals, rays_d):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

    alpha = 1. - torch.exp(-F.relu(raw[..., 3]) * dists)
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha),
                                               1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    acc_map = torch.sum(weights, -1)
    rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, multires, device="cuda:2", include_input=True, input_dims=3, log_sampling=True):
        self.multires = multires
        self.device = device
        self.input_dims = input_dims
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.periodic_fns = [torch.sin, torch.cos]

        self.embed_fns = None
        self.out_dim = None
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.multires - 1
        N_freqs = self.multires

        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs).to(self.device)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs).to(self.device)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        # Implementation according to the official code release
        # (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


class NeRFWrapper:
    def __init__(self, device, initial_lr=0.005):
        self.embedder = Embedder(10, device=device)
        self.embedder_dir = Embedder(4, device=device)
        input_ch = self.embedder.out_dim
        input_ch_views = self.embedder_dir.out_dim
        self.net_chunk = 8192
        self.net = NeRF(D=8, W=256,
                        input_ch=input_ch, output_ch=5, skips=[4],
                        input_ch_views=input_ch_views, use_viewdirs=True).to(device)

        grad_vars = list(self.net.parameters())
        self.optimizer = torch.optim.Adam(params=grad_vars, lr=initial_lr, betas=(0.9, 0.999))
        self.start = 0

        self.N_samples = 64
        self.N_importance = 128
        self.white_bkgd = True
        self.raw_noise_std = 0.0
        self.near = 2.0
        self.far = 6.0

    def load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.start = ckpt['global_step']
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        self.net.load_state_dict(ckpt['network_fn_state_dict'])

    def func(self, inputs, view_dirs):
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embedder.embed(inputs_flat)

        input_dirs = view_dirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = self.embedder_dir.embed(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

        outputs_flat = torch.cat(
            [self.net(
                embedded[i:i + self.net_chunk]) for i in range(0,
                                                               embedded.shape[0], self.net_chunk)], 0)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs

    def cut_batch(self):
        ...

    def render_rays(self, rays):
        rays_num = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # [rays_num, 3] each
        view_directions = rays[:, -3:] if rays.shape[-1] > 8 else None
        bounds = torch.reshape(rays[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=64).to(rays.device)
        z_vals = near * (1. - t_vals) + far * t_vals

        z_vals = z_vals.expand([rays_num, 64])

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
        raw_outputs = self.func(pts, view_directions)
        rgb = volume_rendering(raw_outputs, z_vals, view_directions)
        return rgb

    def render_a_view(self, cam_trans, pose, h=800, w=800, save_name="demo.png", save=True):
        rays_o, rays_d = cam_trans.get_rays(pose)
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = self.near * torch.ones_like(rays_d[..., :1]), self.far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)

        rgb = []
        for i in range(0, rays.shape[0], self.net_chunk):
            with torch.no_grad():
                rgb_mini_batch = self.render_rays(rays[i:i + self.net_chunk])
            rgb.append(rgb_mini_batch)
        rgb_flat = torch.concat(rgb)
        recon_img = torch.reshape(rgb_flat, (h, w, rgb_flat.shape[-1]))
        r_img = (255 * np.clip(recon_img.cpu().numpy(), 0, 1)).astype(np.uint8)
        if save:
            imageio.imwrite(save_name, r_img)
        return r_img

    def render_a_path(self, cam_trans, poses):
        frames = []
        for pose in poses:
            frame = self.render_a_view(cam_trans, pose)
            frames.append(frame)
        imageio.mimsave("spin.gif", frames, "GIF", duration=0.1)


def _test_embedder():
    em = Embedder(10)
    a = torch.Tensor([[3, 4, 6]])
    e = em.embed(a)
    print(e.shape)
    print(e)


if __name__ == "__main__":
    _test_embedder()
