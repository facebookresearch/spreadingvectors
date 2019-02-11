# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import numpy as np
import pdb
from lattices import c_lattices
import unittest
from lattices import Zn_lattice

class TestZnCodec(unittest.TestCase):

    def test_codec(self):
        self.do_test(32, 14)

    def test_codec_rec(self):
        self.do_test(24, 79)

    def do_test(self, dim, r2):
        codec = Zn_lattice.ZnCodec(dim, r2)
        # print("nb atoms", codec.natom)
        rs = np.random.RandomState(123)

        n = 2000
        x = rs.randn(n, dim).astype('float32')
        x /= np.sqrt((x ** 2).sum(1)).reshape(-1, 1)
        quant = codec.quantize(x)

        codes = codec.encode(x)
        x_decoded = codec.decode(codes)

        assert np.all(x_decoded == quant)

        codec2 = Zn_lattice.ZnCodecPy(dim, r2)

        quant2 = codec2.quantize(x)
        assert np.all(quant == quant2)

#####################################################################
# Low-level tests
#####################################################################


swig_ptr = c_lattices.swig_ptr


class BasicTest(unittest.TestCase):

    def test_comb(self):
        assert c_lattices.cvar.comb(2, 1) == 2

    def test_repeats(self):
        rs = np.random.RandomState(123)
        dim = 32
        for i in range(1000):
            vec = np.floor((rs.rand(dim) ** 7) * 3).astype('float32')
            vecs = vec.copy(); vecs.sort()
            repeats = c_lattices.Repeats(dim, swig_ptr(vecs))
            rr = [repeats.repeats.at(i) for i in range(repeats.repeats.size())]
            # print([(r.val, r.n) for r in rr])
            code = repeats.encode(swig_ptr(vec))
            #print(vec, code)
            vec2 = np.zeros(dim, dtype='float32')
            repeats.decode(code, swig_ptr(vec2))
            # print(vec2)
            assert np.all(vec == vec2)


class TestZnSphereCodec(unittest.TestCase):

    def test_codec(self):

        dim = 32
        r2 = 14
        codec = c_lattices.ZnSphereCodec(dim, r2)
        # print("nb atoms", codec.natom)
        rs = np.random.RandomState(123)
        for i in range(1000):
            x = rs.randn(dim).astype('float32')
            ref_c = np.zeros(dim, dtype='float32')
            codec.search(swig_ptr(x), swig_ptr(ref_c))
            code = codec.search_and_encode(swig_ptr(x))
            # print(x, code)
            c = np.zeros(dim, dtype='float32')
            codec.decode(code, swig_ptr(c))
            # print(ref_c, c)


class TestZnSphereCodecRec(unittest.TestCase):

    def test_encode_centroid(self):
        dim = 8
        r2 = 5
        ref_codec = c_lattices.ZnSphereCodec(dim, r2)
        codec = c_lattices.ZnSphereCodecRec(dim, r2)
        # print(ref_codec.nv, codec.nv)
        assert ref_codec.nv == codec.nv
        s = set()
        for i in range(ref_codec.nv):
            c = np.zeros(dim, dtype='float32')
            ref_codec.decode(i, swig_ptr(c))
            code = codec.encode_centroid(swig_ptr(c))
            assert 0 <= code < codec.nv
            s.add(code)
        assert len(s) == codec.nv

    def test_codec(self):
        dim = 16
        r2 = 6
        codec = c_lattices.ZnSphereCodecRec(dim, r2)
        # print("nv=", codec.nv)
        for i in range(codec.nv):
            c = np.zeros(dim, dtype='float32')
            codec.decode(i, swig_ptr(c))
            code = codec.encode_centroid(swig_ptr(c))
            assert code == i



if __name__ == '__main__':
    unittest.main()
