from itertools import product as product
from math import ceil

import torch
import pdb


class PriorBox(object):
    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg["min_sizes"]
        # self.min_sizes -- [[16, 32], [64, 128], [256, 512]]
        self.steps = cfg["steps"]
        # self.steps -- [8, 16, 32]
        self.clip = cfg["clip"]
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]
        # self.feature_maps -- [[32, 32], [16, 16], [8, 8]]
        self.name = "s"

    def forward(self):
        anchors = []
        # (Pdb) for k, f in enumerate(self.feature_maps):print(k, f)
        # 0 [32, 32]
        # 1 [16, 16]
        # 2 [8, 8]
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                # f[0], f[1] -- (32, 32)
                # pp j -- from 0 to 32, i--from 0 32 ?
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        # self.clip -- False
        if self.clip:
            output.clamp_(max=1, min=0)
        # cx.min(), cx.max() -- 0.0160, 1.0080, cy is same as cx
        # s_kx.min(), s_kx.max() -- 0.0640, 2.0480, s_ky is same s_kx

        return output
