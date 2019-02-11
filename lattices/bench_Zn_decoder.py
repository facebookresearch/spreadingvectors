# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from lattices import Zn_lattice
import time
import faiss
import sys

## single-thread benchmark
faiss.omp_set_num_threads(1)


## all data is random, we are not looking at correctness here
if True:
    dim = 32
    r2 = 36
else:
    dim = 24
    r2 = 79

codec = Zn_lattice.ZnCodec(dim, r2)


# set number of queries
nb = 10**6
nq = 1000
k = 1

print("nb=%d nq=%d" % (nb, nq))

# sample queries
rs = np.random.RandomState(123)
xq = rs.randn(nq, dim).astype('float32')


print("init dim=%d r2=%d" % (dim, r2))

codes = rs.randint(1<<31, size=nb).astype('uint64')

print("code_size=%d nv=%d %.2f bits" % (
    codec.code_size, codec.nv, np.log2(codec.nv)))
assert codec.code_size == 8

dis = np.empty((nq, k), dtype='float32')
labels = np.empty((nq, k), dtype='int64')

t0 = time.time()
codec.find_nn(codes, xq)
t1 = time.time()

print ("time for code_size=%d nq=%d nb=%d: %.3f s (%.3f ms/query)" % (
    codec.code_size, nq, nb, t1 - t0,
    (t1 - t0) * 1000 / nq))

index = faiss.IndexPQ(dim, 8, 8)

xb = rs.randn(nb, dim).astype('float32')
print("train")
index.train(xb)
print("add")
index.add(xb)

t0 = time.time()
index.search(xq, 1)
t1 = time.time()

print ("time for IndexPQ code_size=%d nq=%d nb=%d: %.3f s (%.3f ms/query)" % (
    index.pq.code_size, nq, nb, t1 - t0,
    (t1 - t0) * 1000 / nq))
