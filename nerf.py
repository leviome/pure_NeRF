import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hash_utils import get_voxel_vertices, trilinear_interp, RAdam


def sample_pdf(bins, weights, samples_num, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=samples_num)
        u = u.expand(list(cdf.shape[:-1]) + [samples_num])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [samples_num])

    # Invert CDF
    u = u.contiguous().to(cdf)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def volume_rendering(raw, z_vals, rays_d):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

    alpha = 1. - torch.exp(-F.relu(raw[..., 3]) * dists)
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha),
                                               1. - alpha + 1e-10], -1), -1)[:, :-1]

    depth_map = torch.sum(weights * z_vals, -1) / torch.sum(weights, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    acc_map = torch.sum(weights, -1)  # accumulative
    rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, weights, depth_map


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
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1), None


class HashEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution) - torch.log(self.base_resolution)) / (n_levels - 1))

        self.embeddings = nn.ModuleList([nn.Embedding(2 ** self.log2_hashmap_size,
                                                      self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):  # 16
            resolution = torch.floor(self.base_resolution * self.b ** i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(
                x, self.bounding_box,
                resolution, self.log2_hashmap_size)
            # print(keep_mask)
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        keep_mask = keep_mask.sum(dim=-1) == keep_mask.shape[-1]
        return torch.cat(x_embedded_all, dim=-1), keep_mask

    def embed(self, x):
        return self.forward(x)


class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):

        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert 1 <= self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                # result[..., 6] = self.C2[2] * (3.0 * zz - 1)
                # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result

    def embed(self, x):
        return self.forward(x), None


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


class NeRFSmall(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,
                 ):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # sigma
        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma, geo_feat = h[..., 0], h[..., 1:]

        # color
        h = torch.cat([input_views, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # color = torch.sigmoid(h)
        color = h
        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

        return outputs


class NeRFWrapper:
    def __init__(self, device, initial_lr=0.005,
                 use_hash=True,
                 near=2.0,
                 far=6.0):
        self.net_chunk = 8192
        self.device = device
        self.use_hash = use_hash
        if not use_hash:
            self.embedder = Embedder(10, device=device)
            self.embedder_dir = Embedder(4, device=device)
            self.net = NeRF(D=8, W=256,
                            input_ch=self.embedder.out_dim,
                            output_ch=5, skips=[4],
                            input_ch_views=self.embedder_dir.out_dim,
                            use_viewdirs=True).to(device)
            self.net_fine = NeRF(D=8, W=256,
                                 input_ch=self.embedder.out_dim, output_ch=5, skips=[4],
                                 input_ch_views=self.embedder_dir.out_dim,
                                 use_viewdirs=True).to(device)
            # net_fine is for hierarchical volume sampling
            grad_vars = list(self.net.parameters())
            grad_vars += list(self.net_fine.parameters())
            self.optimizer = torch.optim.Adam(params=grad_vars, lr=initial_lr, betas=(0.9, 0.999))
        else:
            bounding_box = [[-4.0182, -4.0080, -3.3300],
                            [4.0072, 4.0174, 3.3384]]
            bbox = torch.Tensor(bounding_box).to(device)
            self.embedder = HashEmbedder(bbox).to(device)
            embedding_params = list(self.embedder.parameters())
            self.embedder_dir = SHEncoder().to(device)
            self.net = NeRFSmall(num_layers=2,
                                 hidden_dim=64,
                                 geo_feat_dim=15,
                                 num_layers_color=3,
                                 hidden_dim_color=64,
                                 input_ch=self.embedder.out_dim,
                                 input_ch_views=self.embedder_dir.out_dim).to(device)

            grad_vars = list(self.net.parameters())

            self.net_fine = NeRFSmall(num_layers=2,
                                      hidden_dim=64,
                                      geo_feat_dim=15,
                                      num_layers_color=3,
                                      hidden_dim_color=64,
                                      input_ch=self.embedder.out_dim,
                                      input_ch_views=self.embedder_dir.out_dim).to(device)
            grad_vars += list(self.net_fine.parameters())

            self.optimizer = RAdam([
                {'params': grad_vars, 'weight_decay': 1e-6},
                {'params': embedding_params, 'eps': 1e-15}
            ], lr=0.01, betas=(0.9, 0.99))

        self.start = 0

        self.N_samples = 64
        self.N_importance = 128
        self.white_bkgd = True
        self.raw_noise_std = 0.0
        self.near = near
        self.far = far

    def load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.start = ckpt['global_step']
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        self.net.load_state_dict(ckpt['net_state_dict'])
        self.net_fine.load_state_dict(ckpt['net_fine_state_dict'])

        if self.use_hash:
            self.embedder.load_state_dict(ckpt["hash_em"])

    def func(self, inputs, view_dirs, use_fine=False):
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded, keep_mask = self.embedder.embed(inputs_flat)

        input_dirs = view_dirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs, _ = self.embedder_dir.embed(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
        _net = self.net if not use_fine else self.net_fine
        outputs_flat = torch.cat(
            [_net(
                embedded[i:i + self.net_chunk]) for i in range(0,
                                                               embedded.shape[0], self.net_chunk)], 0)
        if keep_mask is not None:
            outputs_flat[~keep_mask, -1] = 0
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

        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(rays.device)
        z_vals = near * (1. - t_vals) + far * t_vals

        z_vals = z_vals.expand([rays_num, self.N_samples])
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        raw_outputs = self.func(pts, view_directions)
        rgb, weights, _ = volume_rendering(raw_outputs, z_vals, rays_d)

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], self.N_importance)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        # [N_rays, N_samples + N_importance, 3]

        raw_outputs_fine = self.func(pts, view_directions, use_fine=True)
        rgb_fine, _, depth = volume_rendering(raw_outputs_fine, z_vals, rays_d)
        return rgb, rgb_fine, depth

    def render_a_view(self, cam_trans, pose, h=400, w=400, save_name="demo.png", save=True):
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
                _, rgb_mini_batch, depth = self.render_rays(rays[i:i + self.net_chunk])
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
    e, _ = em.embed(a)
    print(e.shape)
    print(e)


def _test_hash():
    device = "cuda:3"

    bounding_box = [[-4.0182, -4.0080, -3.3300],
                    [4.0072, 4.0174, 3.3384]]
    bbox = torch.Tensor(bounding_box).to(device)
    print(bbox)
    hash_embed = HashEmbedder(bbox)
    hash_embed = hash_embed.to(device)
    x = torch.randn((1, 3))
    print(x)
    # x = torch.tensor([[83, 3, 3]])
    x = x.to(device)
    y, keep_mask = hash_embed(x)
    print(y)
    print(y.shape)

    print(keep_mask)


if __name__ == "__main__":
    # _test_embedder()
    _test_hash()
