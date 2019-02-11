# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import numpy as np
from lattices.Zn_lattice import ZnCodec


class Quantizer:
    def __init__(self):
        self.requires_train = False
        self.asymmetric = True

    def train(self, x):
        "do nothing by default"
        pass

    def quantize(self, x):
        "return closest quantized vector"
        raise NotImplementedError("Function is not implemented")

    def __call__(self, x):
        return self.quantize(x)

class Zn(Quantizer):
    def __init__(self, r2, d):
        super(Zn, self).__init__()
        self.r2 = r2
        self.r = np.sqrt(self.r2)
        self.d = d
        self.codec = ZnCodec(self.d, self.r2)
        ntot = self.codec.nv
        self.bits = int(np.ceil(np.log2(float(ntot))))

    def quantize(self, x):
        if not np.all(np.abs(np.linalg.norm(x, axis=1) - 1) < 1e-5):
            print("WARNING: Vectors were not L2 normalized in Zn")

        return self.codec.quantize(self.r * x) / self.r


class Identity(Quantizer):
    def __init__(self):
        super(Identity, self).__init__()
        self.bits = None

    def quantize(self, x):
        assert x.ndim == 2
        self.bits = 8 * x.nbytes // x.shape[0]
        return x

try:
    import faiss
except ImportError:
    faiss = None


def to_binary(x):
    n, d = x.shape
    assert d % 8 == 0
    if faiss is None:
        return ((x >= 0).reshape(n, d // 8, 8) *
                (1 << np.arange(8)).astype('uint8')).sum(2)
    else:
        y = np.empty((n, d // 8), dtype='uint8')
        faiss.real_to_binary(n * d, faiss.swig_ptr(x), faiss.swig_ptr(y))
        return y

class OPQ(Quantizer):
    def __init__(self, nbytes, d):
        super(OPQ, self).__init__()
        self.dim = d
        self.requires_train = True
        self.bits = nbytes * 8
        self.index = faiss.index_factory(
            self.dim, "OPQ%d_%d,PQ%d" % (nbytes, nbytes * 8, nbytes))

    def train(self, x):
        self.index.train(x)

    def quantize(self, x):
        if not self.index.is_trained:
            print("WARNING: training OPQ inside the quantize() function")
            self.train(x[:10000])
        print("Adding vectors")
        self.index.add(x)
        return self.index.reconstruct_n(0, x.shape[0])


class Binary(Quantizer):
    def __init__(self, d):
        super(Binary, self).__init__()
        self.bits = None
        self.asymmetric = False
        self.dim = d
        self.bits = (d + 7) // 8 * 8

    def quantize(self, x):
        assert x.ndim == 2
        return np.sign(x)
        # return to_binary(x)

def getQuantizer(snippet, d):
    if snippet.startswith("zn_"):
        r2 = int(snippet.split("_")[1])
        return Zn(r2, d)
    elif snippet == "binary":
        return Binary(d)
    elif snippet == "none":
        return Identity(d)
    elif snippet.startswith("opq_"):
        nbytes = int(snippet.split("_")[1]) // 8
        return OPQ(nbytes, d)
    else:
        raise NotImplementedError("Quantizer not implemented")
