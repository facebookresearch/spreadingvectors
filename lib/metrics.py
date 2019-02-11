# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division, print_function

try:
    import faiss
    hasfaiss = True
except:
    hasfaiss = False

import torch
import shutil
from os.path import join
from .quantizers import getQuantizer
from .net import forward_pass
import numpy as np
import time


#######################################################
# Nearest neighbor search functions
#######################################################


def get_nearestneighbors_faiss(xq, xb, k, device, needs_exact=True, verbose=False):
    assert device in ["cpu", "cuda"]

    if verbose:
        print("Computing nearest neighbors (Faiss)")

    if needs_exact or device == 'cuda':
        index = faiss.IndexFlatL2(xq.shape[1])
    else:
        index = faiss.index_factory(xq.shape[1], "HNSW32")
        index.hnsw.efSearch = 64
    if device == 'cuda':
        index = faiss.index_cpu_to_all_gpus(index)

    start = time.time()
    index.add(xb)
    _, I = index.search(xq, k)
    if verbose:
        print("  NN search (%s) done in %.2f s" % (
            device, time.time() - start))

    return I


def cdist2(A, B):
    return  (A.pow(2).sum(1, keepdim = True)
             - 2 * torch.mm(A, B.t())
             + B.pow(2).sum(1, keepdim = True).t())

def top_dist(A, B, k):
    return cdist2(A, B).topk(k, dim=1, largest=False, sorted=True)[1]

def get_nearestneighbors_torch(xq, xb, k, device, needs_exact=False, verbose=False):
    if verbose:
        print("Computing nearest neighbors (torch)")

    assert device in ["cpu", "cuda"]
    start = time.time()
    xb, xq = torch.from_numpy(xb), torch.from_numpy(xq)
    xb, xq = xb.to(device), xq.to(device)
    bs = 500
    I = torch.cat([top_dist(xq[i*bs:(i+1)*bs], xb, k)
                   for i in range(xq.size(0) // bs)], dim=0)
    if verbose:
        print("  NN search done in %.2f s" % (time.time() - start))
    I = I.cpu()
    return I.numpy()

if hasfaiss:
    get_nearestneighbors = get_nearestneighbors_faiss
else:
    get_nearestneighbors = get_nearestneighbors_torch




#######################################################
# Evaluation metrics
#######################################################


def sanitize(x):
    return np.ascontiguousarray(x, dtype='float32')


def evaluate(net, xq, xb, gt, quantizers, best_key, device=None,
             trainset=None):
    net.eval()
    if device is None:
        device = next(net.parameters()).device.type
    xqt = forward_pass(net, sanitize(xq), device=device)
    xbt = forward_pass(net, sanitize(xb), device=device)
    if trainset is not None:
        trainset = forward_pass(net, sanitize(trainset), device=device)
    nq, d = xqt.shape
    res = {}
    score = 0
    for quantizer in quantizers:
        qt = getQuantizer(quantizer, d)

        qt.train(trainset)
        xbtq = qt(xbt)
        if not qt.asymmetric:
            xqtq = qt(xqt)
            I = get_nearestneighbors(xqtq, xbtq, 100, device)
        else:
            I = get_nearestneighbors(xqt, xbtq, 100, device)

        print("%s\t nbit=%3d: " % (quantizer, qt.bits), end=' ')

        # compute 1-recall at ranks 1, 10, 100 (comparable with
        # fig 5, left of the paper)
        recalls = []
        for rank in 1, 10, 100:
            recall = (I[:, :rank] == gt[:, :1]).sum() / float(nq)
            key = '%s,rank=%d' % (quantizer, rank)
            if key == best_key:
                score = recall
            recalls.append(recall)
            print('%.4f' % recall, end=" ")
        res[quantizer] = recalls
        print("")

    return res, score



class ValidationFunction:

    def __init__(self, xq, xb, gt, checkpoint_dir, validation_key,
                 quantizers=[]):
        assert type(quantizers) == list
        self.xq = xq
        self.xb = xb
        self.gt = gt
        self.checkpoint_dir = checkpoint_dir
        self.best_key = validation_key
        self.best_score = 0
        self.quantizers = quantizers

    def __call__(self, net, epoch, args, all_logs):
        """
        Evaluates the current state of the network without
        and with quantization and stores a checkpoint.
        """
        print("Valiation at epoch %d" % epoch)
        # also store current state of network + arguments
        res, score = evaluate(net, self.xq, self.xb, self.gt,
                              self.quantizers, self.best_key)
        all_logs[-1]['val'] = res
        if self.checkpoint_dir:
            fname = join(self.checkpoint_dir, "checkpoint.pth")
            print("storing", fname)
            torch.save({
                'state_dict': net.state_dict(),
                'epoch': epoch,
                'args': args,
                'logs': all_logs
            }, fname)
            if score > self.best_score:
                print("%s score improves (%g > %g), keeping as best"  % (
                    self.best_key, score, self.best_score))
                self.best_score = score
                shutil.copyfile(fname, fname + '.best')

        return res
