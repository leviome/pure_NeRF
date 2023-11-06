import numpy as np
import torch


def pose_spherical_v2(theta, phi, r):
    phi, th = phi / 180. * np.pi, theta / 180. * np.pi
    pose = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, r], [0, 0, 0, 1]]).float()
    rotate_phi = torch.Tensor([[1, 0, 0, 0],
                               [0, np.cos(phi), -np.sin(phi), 0],
                               [0, np.sin(phi), np.cos(phi), 0],
                               [0, 0, 0, 1]]).float()
    rotate_theta = torch.Tensor([[np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0],
                                 [np.sin(th), 0, np.cos(th), 0], [0, 0, 0, 1]]).float()
    trans = torch.Tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    pose = trans @ (rotate_theta @ (rotate_phi @ pose))
    return pose


class CamTrans:
    def __init__(self, focal, h=800, w=800, device="cuda:2"):
        self.focal_x = focal
        self.focal_y = focal
        self.offset_x = w / 2
        self.offset_y = h / 2
        self.h, self.w = h, w
        self.device = device

        self.intri = [[self.focal_x, 0, self.offset_x],
                      [0, self.focal_y, self.offset_y],
                      [0, 0, 1]]

        # to load
        self.extri = None
        self.R = None
        self.T = None

        # for random rays
        coords = torch.stack(torch.meshgrid(torch.linspace(0, h - 1, h).to(device),
                                            torch.linspace(0, w - 1, w).to(device)),
                             -1)  # (H, W, 2)

        self.coords = torch.reshape(coords, [-1, 2]).to(device)  # (H * W, 2)

    def get_rays(self, c2w):
        c2w = torch.Tensor(c2w).to(self.device)
        i, j = torch.meshgrid(torch.linspace(0, self.w - 1, self.w),
                              torch.linspace(0, self.h - 1, self.h))
        # pytorch's meshgrid has indexing='ij'

        i = i.t()
        j = j.t()

        # directions
        dirs = torch.stack([(i - self.offset_x) / self.focal_x,
                            -(j - self.offset_y) / self.focal_y,
                            -torch.ones_like(i)], -1).to(self.device)

        # camera to world
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3, -1].expand(rays_d.shape)

        return rays_o, rays_d

    def random_rays(self, rays_o, rays_d, target, n_rand=1024):
        select_indices = np.random.choice(self.coords.shape[0], size=[n_rand], replace=False)  # (N_rand,)
        select_coords = self.coords[select_indices].long()  # (n_rand, 2)

        batch_rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (n_rand, 3)
        batch_rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (n_rand, 3)

        target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (n_rand, 3)
        return batch_rays_o, batch_rays_d, target_s

    def load_extri(self, extri):
        self.extri = extri
        self.R = self.extri[:3, :3]
        self.T = self.extri[:3, -1]

    def world2cam(self, p):
        return np.dot(p - self.T, self.R)

    def cam2screen(self, p):
        """
                          x                                y
            x_im = f_x * --- + offset_x      y_im = f_y * --- + offset_y
                          z                                z
        """
        x, y, z = p
        return [-x * self.focal_x / z + self.offset_x, y * self.focal_y / z + self.offset_y]

    def world2screen(self, p):
        return self.cam2screen(self.world2cam(p))
