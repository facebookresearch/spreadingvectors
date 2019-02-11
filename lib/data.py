# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from os.path import join
import numpy as np
from .metrics import get_nearestneighbors, sanitize


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def getBasedir(s):
    paths = {
        "bigann": "/datasets01/simsearch/041218/bigann",
        "deep1b": "/datasets01/simsearch/041218/deep1b"
    }

    return paths[s]


def load_deep1b(device, size = 10 ** 6, test=True, qsize=10 ** 5):
    basedir = getBasedir("deep1b") + '/'
    xt = mmap_fvecs(basedir + 'learn.fvecs')
    if test:
        xb = mmap_fvecs(basedir + 'base.fvecs')
        xq = mmap_fvecs(basedir + 'deep1B_queries.fvecs')
        xb = xb[:size]

        gt = ivecs_read(basedir + 'deep%s_groundtruth.ivecs' % (
            '1M' if size == 10 ** 6 else
            '10M' if size == 10 ** 7 else
            '100M' if size == 10 ** 8 else
            '1B' if size == 10 ** 9 else 1/0
        ))
    else:
        xb = xt[:size]
        xq = xt[size:size+qsize]
        xt = xt[size+qsize:]

    xb, xq = sanitize(xb), sanitize(xq)
    if not test:
        gt = get_nearestneighbors(xq, xb, 100, device)

    return xt, xb, xq, gt


def load_bigann(device, size = 10 ** 6, test=True, qsize=10 ** 5):
    basedir = getBasedir("bigann")

    dbsize = int(size / 10 ** 6)
    xt = mmap_bvecs(join(basedir, 'bigann_learn.bvecs'))
    if test:
        xb = mmap_bvecs(join(basedir, 'bigann_base.bvecs'))
        xq = mmap_bvecs(join(basedir, 'bigann_query.bvecs'))
        # trim xb to correct size
        xb = xb[:dbsize * 1000 * 1000]
        gt = ivecs_read(join(basedir, 'gnd/idx_%dM.ivecs' % dbsize))
    else:
        xb = xt[:size]
        xq = xt[size:size+qsize]
        xt = xt[size+qsize:]

    xb, xq = sanitize(xb), sanitize(xq)
    if not test:
        gt = get_nearestneighbors(xq, xb, 100, device)

    return xt, xb, xq, gt


def load_dataset(name, device, size=10**6, test=True):
    if name == "bigann":
        return load_bigann(device, size, test)
    elif name == "deep1b":
        return load_deep1b(device, size, test)
