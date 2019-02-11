# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class StraightThroughQuantizer(nn.Module):
    def __init__(self, qt):
        super(StraightThroughQuantizer, self).__init__()
        self.quantizer = qt

    def __call__(self, x):
        if type(x) is np.ndarray:
            return self.quantizer.quantize(x)
        else:
            return super(StraightThroughQuantizer, self).__call__(x)

    def forward(self, x):
        q_x = torch.from_numpy(self.quantizer.quantize(x.detach().cpu().numpy()))
        q_x = q_x.to(x.device)

        # Hack to get the straight thru estimator
        return q_x + x - x.detach()


def forward_pass(net, xall, bs=128, device=None):
    if device is None:
        device = next(net.parameters()).device
    xl_net = []
    for i0 in range(0, xall.shape[0], bs):
        x = torch.from_numpy(xall[i0:i0 + bs])
        x = x.to(device)
        xl_net.append(net(x).data.cpu().numpy())

    return np.vstack(xl_net)


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)
