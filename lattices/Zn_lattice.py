# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np

try:
    from lattices import c_lattices
    swig_ptr = c_lattices.swig_ptr
except ImportError:
    c_lattices = None


class Comb:
    """ a Pascal triangle """
    def __init__(self, npas=100):
        pascal = [[1] for i in range(npas)]
        for i in range(npas):
            for j in range(1, i + 1):
                pascal[i].append(pascal[i - 1][j] + pascal[i - 1][j - 1])
            pascal[i].append(0)
        self.pascal = pascal

    def __call__(self, n, k):
        return self.pascal[n][k] if k <= n else 0

# function to compute a binomial coefficient
if c_lattices is None:
    comb = Comb()
else:
    comb = c_lattices.cvar.comb

def count_comb(x):
    """count number of distinct permutations of an array that contains
    duplicate values"""
    x = np.array(x)
    n = len(x)
    accu = 1
    for v in np.unique(x):
        nv = int((x == v).sum())
        accu *= int(comb(n, nv))
        n -= nv
    # bits used for signs
    accu *= 2 ** int((x != 0).sum())
    return accu


def sum_of_sq(total, v, n):
    """find all positive integer vectors of size n:
    - whose squared elements sum to total
    - maximium value is v
    """

    if total < 0:
        return []
    elif total == 0:
        return [[0] * n]
    elif n == 1:
        while v * v > total:
            v -= 1
        if v * v == total:
            return [[v]]
        else:
            return []
    else:
        res = []
        for vi in range(v, -1, -1):
            res += [[vi] + vv for vv in
                    sum_of_sq(total - vi * vi, vi, n - 1)]
        return res


def compute_atoms(d, r2):
    """Find atoms that define the Zn sphere of dimension d and squared
    radius r2"""
    v = int(1 + np.sqrt(r2)) # max value of a component
    if c_lattices is None:
        atoms = sum_of_sq(r2, v, d)
        return np.array(atoms)
    else:
        atoms = c_lattices.sum_of_sq(r2, v, d)
        return c_lattices.vector_to_array(atoms).reshape(-1, d)



class ZnCodecC:

    def __init__(self, d, r2):
        self.znc = c_lattices.ZnSphereCodec(d, r2)
        atoms = c_lattices.vector_to_array(self.znc.voc)
        atoms = atoms.reshape(-1, d)
        # recompute instead of using self.znc.nv because it is limited
        # to 64 bit
        self.nv = sum([count_comb(atom) for atom in atoms])
        self.code_size = self.znc.code_size

        if d & (d - 1) == 0:
            # d is a power of 2. Then we can use a ZnSphereCodecRec as
            # codec (faster for decoding)
            self.znc_rec = c_lattices.ZnSphereCodecRec(d, r2)
        else:
            self.znc_rec = None


    def quantize(self, x):
        x = np.ascontiguousarray(x, dtype='float32')
        n, d = x.shape
        assert d == self.znc.dim
        c = np.empty((n, d), dtype='float32')
        dps = np.empty(n, dtype='float32')
        self.znc.search_multi(n,
                              swig_ptr(x), swig_ptr(c),
                              swig_ptr(dps))
        return c

    def encode(self, x):
        assert self.nv < 2 ** 64
        n, d = x.shape
        assert d == self.znc.dim
        codes = np.empty(n, dtype='uint64')
        if not self.znc_rec:
            self.znc.encode_multi(n, swig_ptr(x),
                                  swig_ptr(codes))
        else:
            # first quantizer then encode
            centroids = self.quantize(x)
            self.znc_rec.encode_multi(n, swig_ptr(centroids),
                                      swig_ptr(codes))
        return codes

    def decode(self, codes):
        n, = codes.shape
        x = np.empty((n, self.znc.dim), dtype='float32')
        decoder = self.znc_rec or self.znc
        decoder.decode_multi(n, swig_ptr(codes),
                            swig_ptr(x))
        return x

    def find_nn(self, codes, xq):
        """ find the nearest code of each vector of xq
        (returns dot products, not distances)
        """
        assert self.nv < 2 ** 64
        nc, = codes.shape
        nq, d = xq.shape
        assert d == self.znc.dim
        ids = np.empty(nq, dtype='int64')
        dis = np.empty(nq, dtype='float32')
        decoder = self.znc_rec or self.znc
        decoder.find_nn(nc, swig_ptr(codes), nq, swig_ptr(xq),
                        swig_ptr(ids), swig_ptr(dis))

        return ids, dis



class ZnCodecPy:

    def __init__(self, d, r2):
        self.atoms = compute_atoms(d, r2)
        self.atoms = np.sort(self.atoms, axis=1)
        self.nv = sum([count_comb(atom) for atom in self.atoms])

    def quantize(self, x):
        n, d = x.shape
        assert d == self.atoms.shape[1]
        x_abs = np.abs(x)
        x_mod = np.sort(x_abs, axis=1)
        x_order = np.argsort(np.argsort(x_abs, axis=1), axis=1)
        matches = np.argmax(np.dot(x_mod, self.atoms.T), axis=1)
        x_recons = self.atoms[matches]
        q_abs = x_recons[np.tile(np.arange(n).reshape(-1, 1), d), x_order]
        q = q_abs * np.sign(x)

        return q.astype('float32')

if c_lattices is None:
    ZnCodec = ZnCodecPy
else:
    ZnCodec = ZnCodecC
