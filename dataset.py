import os
import json
import numpy as np
import cv2
import imageio


class DataForNeRF:
    def __init__(self, root, half_res=True):
        self.root = root
        self.splits = ['train', 'val', 'test']
        self.metas = {}
        for s in self.splits:
            with open(os.path.join(root, 'transforms_{}.json'.format(s)), 'r') as fp:
                self.metas[s] = json.load(fp)
        tmp = cv2.imread(os.path.join(root, self.metas['train']['frames'][0]['file_path'] + '.png'))
        h, w, c = tmp.shape
        self.focal = .5 * w / np.tan(.5 * float(self.metas["train"]["camera_angle_x"]))
        self.h, self.w = h, w

        self.poses = None
        self.images = None
        self.i_split = None

        self.load_all(half_res)

    def load_all(self, half_res):
        all_imgs = []
        all_poses = []
        counts = [0]
        for s in self.splits:
            meta = self.metas[s]
            imgs = []
            poses = []
            for frame in meta['frames']:
                fname = os.path.join(self.root, frame['file_path'] + '.png')
                imgs.append(imageio.v2.imread(fname))
                poses.append(np.array(frame['transform_matrix']))  # extri params
            imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)
        self.i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
        images = np.concatenate(all_imgs, 0)

        self.poses = np.concatenate(all_poses, 0)
        if half_res:
            self.h = self.h // 2
            self.w = self.w // 2
            self.focal = self.focal / 2.

            images_half_res = np.zeros((images.shape[0], self.h, self.w, 4))
            for i, img in enumerate(images):
                images_half_res[i] = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
            images = images_half_res

        self.images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
