# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
from lib.data import load_dataset
import time
import argparse
import numpy as np
from torch import nn, optim
from lib.metrics import ValidationFunction, get_nearestneighbors, sanitize
from lib.net import Normalize, forward_pass, StraightThroughQuantizer
from lib.quantizers import Zn
import torch.nn.functional as F
import torch
import itertools


def repeat(l, r):
    return list(itertools.chain.from_iterable(itertools.repeat(x, r) for x in l))


def pairwise_NNs_inner(x):
    """
    Pairwise nearest neighbors for L2-normalized vectors.
    Uses Torch rather than Faiss to remain on GPU.
    """
    # parwise dot products (= inverse distance)
    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[::(n+1)].fill_(-1)  # Trick to fill diagonal with -1
    _, I = torch.max(dots, 1)  # max inner prod -> min distance
    return I


def triplet_optimize(xt, gt_nn, net, args, val_func):
    """
    train a triplet loss on the training set xt (a numpy array)
    gt_nn:    ground-truth nearest neighbors in input space
    net:      network to optimize
    args:     various runtime arguments
    val_func: callback called periodically to evaluate the network
    """

    lr_schedule = [float(x.rstrip().lstrip()) for x in args.lr_schedule.split(",")]
    assert args.epochs % len(lr_schedule) == 0
    lr_schedule = repeat(lr_schedule, args.epochs // len(lr_schedule))
    print("Lr schedule", lr_schedule)

    N, kpos = gt_nn.shape

    if args.quantizer_train != "":
        assert args.quantizer_train.startswith("zn_")
        r2 = int(args.quantizer_train.split("_")[1])
        qt = StraightThroughQuantizer(Zn(r2))
    else:
        qt = lambda x: x

    xt_var = torch.from_numpy(xt).to(args.device)

    # prepare optimizer
    optimizer = optim.SGD(net.parameters(), lr_schedule[0], momentum=args.momentum)
    pdist = nn.PairwiseDistance(2)

    all_logs = []
    for epoch in range(args.epochs):
        # Update learning rate
        args.lr = lr_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

        t0 = time.time()

        # Sample positives for triplet
        rank_pos = np.random.choice(kpos, size=N)
        positive_idx = gt_nn[np.arange(N), rank_pos]

        # Sample negatives for triplet
        net.eval()
        print("  Forward pass")
        xl_net = forward_pass(net, xt, 1024)
        print("  Distances")
        I = get_nearestneighbors(xl_net, qt(xl_net), args.rank_negative, args.device, needs_exact=False)
        negative_idx = I[:, -1]

        # training pass
        print("  Train")
        net.train()
        avg_triplet, avg_uniform, avg_loss = 0, 0, 0
        offending = idx_batch = 0

        # process dataset in a random order
        perm = np.random.permutation(N)

        t1 = time.time()

        for i0 in range(0, N, args.batch_size):
            i1 = min(i0 + args.batch_size, N)
            n = i1 - i0

            data_idx = perm[i0:i1]

            # anchor, positives, negatives
            ins = xt_var[data_idx]
            pos = xt_var[positive_idx[data_idx]]
            neg = xt_var[negative_idx[data_idx]]

            # do the forward pass (+ record gradients)
            ins, pos, neg = net(ins), net(pos), net(neg)
            pos, neg = qt(pos), qt(neg)

            # triplet loss
            per_point_loss = pdist(ins, pos) - pdist(ins, neg)
            per_point_loss = F.relu(per_point_loss)
            loss_triplet = per_point_loss.mean()
            offending += torch.sum(per_point_loss.data > 0).item()

            # entropy loss
            I = pairwise_NNs_inner(ins.data)
            distances = pdist(ins, ins[I])
            loss_uniform = - torch.log(n * distances).mean()

            # combined loss
            loss = loss_triplet + args.lambda_uniform * loss_uniform

            # collect some stats
            avg_triplet += loss_triplet.data.item()
            avg_uniform += loss_uniform.data.item()
            avg_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx_batch += 1

        avg_triplet /= idx_batch
        avg_uniform /= idx_batch
        avg_loss /= idx_batch

        logs = {
            'epoch': epoch,
            'loss_triplet': avg_triplet,
            'loss_uniform': avg_uniform,
            'loss': avg_loss,
            'offending': offending,
            'lr': args.lr
        }
        all_logs.append(logs)

        t2 = time.time()
        # maybe perform a validation run
        if (epoch + 1) % args.val_freq == 0:
            logs['val'] = val_func(net, epoch, args, all_logs)

        t3 = time.time()

        # synthetic logging
        print ('epoch %d, times: [hn %.2f s epoch %.2f s val %.2f s]'
               ' lr = %f'
               ' loss = %g = %g + lam * %g, offending %d' % (
            epoch, t1 - t0, t2 - t1, t3 - t2,
            args.lr,
            avg_loss, avg_triplet, avg_uniform, offending
        ))

        logs['times'] = (t1 - t0, t2 - t1, t3 - t2)

    return all_logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('dataset options')
    aa("--database", default="deep1b") # can be "bigann", "deep1b" or "*.fvecs"
    aa("--size_base", type=int, default=int(1e6),
       help="size of evaluation dataset")
    aa("--num_learn", type=int, default=int(5e5),
       help="nb of learning vectors")

    group = parser.add_argument_group('Model hyperparameters')
    aa("--dint", type=int, default=1024,
       help="size of hidden states")
    aa("--dout", type=int, default=16,
       help="output dimension")
    aa("--lambda_uniform", type=float, default=0.05,
       help="weight of the uniformity loss")

    group = parser.add_argument_group('Training hyperparameters')
    aa("--batch_size", type=int, default=64)
    aa("--epochs", type=int, default=160)
    aa("--momentum", type=float, default=0.9)
    aa("--rank_positive", type=int, default=10,
       help="this number of vectors are considered positives")
    aa("--rank_negative", type=int, default=50,
       help="these are considered negatives")

    group = parser.add_argument_group('Computation params')
    aa("--seed", type=int, default=1234)
    aa("--checkpoint_dir", type=str, default="",
       help="checkpoint directory")
    aa("--init_name", type=str, default="",
       help="checkpoint to load from")
    aa("--save_best_criterion", type=str, default="",
       help="for example r2=4,rank=10")
    aa("--quantizer_train", type=str, default="")
    aa("--lr_schedule", type=str, default="0.1,0.1,0.05,0.01")
    aa("--device", choices=["cuda", "cpu", "auto"], default="auto")
    aa("--val_freq", type=int, default=10,
       help="frequency of validation calls")
    aa("--validation_quantizers", type=str, default="",
       help="r2 values to try in validation")

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Radiuses that correspond to 16, 32 and 64 bits for Zn
    radiuses = {
        16: [4, 21, 200],
        24: [3, 10, 79],
        32: [3, 8, 36],
        40: [2, 7, 24],
    }
    # Validation quantizers default to Zn
    if args.validation_quantizers == "":
        args.validation_quantizers = ["zn_%d" % x for x in radiuses[args.dout]]
    else:
        args.validation_quantizers = [x.rstrip().lstrip() for x in args.validation_quantizers.split(",")]
    # Default save_best is 64 bits for Zn
    if args.save_best_criterion == "":
        args.save_best_criterion = "zn_%d,rank=10" % radiuses[args.dout][-1]
    print(args)

    print ("load dataset %s" % args.database)
    (xt, xb, xq, gt) = load_dataset(args.database, args.device, size=args.size_base, test=False)

    print ("keeping %d/%d training vectors" % (args.num_learn, xt.shape[0]))
    xt = sanitize(xt[:args.num_learn])

    print ("computing training ground truth")
    xt_gt = get_nearestneighbors(xt, xt, args.rank_positive, device=args.device)

    print ("build network")

    dim = xb.shape[1]
    dint, dout = args.dint, args.dout

    net = nn.Sequential(
        nn.Linear(in_features=dim, out_features=dint, bias=True),
        nn.BatchNorm1d(dint),
        nn.ReLU(),
        nn.Linear(in_features=dint, out_features=dint, bias=True),
        nn.BatchNorm1d(dint),
        nn.ReLU(),
        nn.Linear(in_features=dint, out_features=dout, bias=True),
        Normalize()
    )

    if args.init_name != '':
        print ("loading state from %s" % args.init_name)
        ckpt = torch.load(args.init_name)
        net.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']

    net.to(args.device)

    val = ValidationFunction(xq, xb, gt, args.checkpoint_dir,
                             validation_key=args.save_best_criterion,
                             quantizers=args.validation_quantizers)

    all_logs = triplet_optimize(xt, xt_gt, net, args, val)
