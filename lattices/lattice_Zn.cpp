/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "lattice_Zn.h"

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>

#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>



void VectorCodec::encode_multi(size_t n, const float *c,
                               uint64_t * codes) const
{
#pragma omp parallel if (n > 1000)
    {
#pragma omp for
        for(int i = 0; i < n; i++) {
            codes[i] = encode(c + i * dim);
        }
    }
}


void VectorCodec::decode_multi(size_t n, const uint64_t * codes,
                               float *c) const
{
#pragma omp parallel if (n > 1000)
    {
#pragma omp for
        for(int i = 0; i < n; i++) {
            decode(codes[i], c + i * dim);
        }
    }
}

void VectorCodec::find_nn (
                  size_t nc, const uint64_t * codes,
                  size_t nq, const float *xq,
                  long *labels, float *distances)
{
    for (long i = 0; i < nq; i++) {
        distances[i] = -1e20;
        labels[i] = -1;
    }

    float c[dim];
    for(long i = 0; i < nc; i++) {
        uint64_t code = codes[nc];
        decode(code, c);
        for (long j = 0; j < nq; j++) {
            const float *x = xq + j * dim;
            float dis = fvec_inner_product(x, c, dim);
            if (dis > distances[j]) {
                distances[j] = dis;
                labels[j] = i;
            }
        }
    }

}


/**********************************************************
 * ZnSphereSearch
 **********************************************************/


ZnSphereSearch::ZnSphereSearch(int dim, int r2): dimS(dim), r2(r2) {
    voc = sum_of_sq(r2, int(ceil(sqrt(r2)) + 1), dim);
    natom = voc.size() / dim;
}

float ZnSphereSearch::search(const float *x, float *c) const {
    float tmp[dimS * 2];
    int tmp_int[dimS];
    return search(x, c, tmp, tmp_int);
}

float ZnSphereSearch::search(const float *x, float *c,
                             float *tmp, // size 2 *dim
                             int *tmp_int, // size dim
                             int *ibest_out
                             ) const {
    int dim = dimS;
    assert (natom > 0);
    int *o = tmp_int;
    float *xabs = tmp;
    float *xperm = tmp + dim;

    // argsort
    for (int i = 0; i < dim; i++) {
        o[i] = i;
        xabs[i] = fabsf(x[i]);
    }
    std::sort(o, o + dim, [xabs](int a, int b) {
            return xabs[a] > xabs[b];
        });
    for (int i = 0; i < dim; i++) {
        xperm[i] = xabs[o[i]];
    }
    // find best
    int ibest = -1;
    float dpbest = -100;
    for (int i = 0; i < natom; i++) {
        float dp = fvec_inner_product (voc.data() + i * dim, xperm, dim);
        if (dp > dpbest) {
            dpbest = dp;
            ibest = i;
        }
    }
    // revert sort
    const float *cin = voc.data() + ibest * dim;
    for (int i = 0; i < dim; i++) {
        c[o[i]] = copysignf (cin[i], x[o[i]]);
    }
    if (ibest_out)
        *ibest_out = ibest;
    return dpbest;
}

void ZnSphereSearch::search_multi(int n, const float *x,
                                  float *c_out,
                                  float *dp_out) {
#pragma omp parallel if (n > 1000)
    {
#pragma omp for
        for(int i = 0; i < n; i++) {
            dp_out[i] = search(x + i * dimS, c_out + i * dimS);
        }
    }
}


/**********************************************************
 * ZnSphereCodec
 **********************************************************/

ZnSphereCodec::ZnSphereCodec(int dim, int r2):
    ZnSphereSearch(dim, r2),
    VectorCodec(dim)
{
    nv = 0;
    for (int i = 0; i < natom; i++) {
        Repeats repeats(dim, &voc[i * dim]);
        CodeSegment cs(repeats);
        cs.c0 = nv;
        Repeat &br = repeats.repeats.back();
        cs.signbits = br.val == 0 ? dim - br.n : dim;
        code_segments.push_back(cs);
        nv += repeats.count() << cs.signbits;
    }

    uint64_t nvx = nv;
    code_size = 0;
    while (nvx > 0) {
        nvx >>= 8;
        code_size++;
    }
}

uint64_t ZnSphereCodec::search_and_encode(const float *x) const {
    float tmp[dim * 2];
    int tmp_int[dim];
    int ano; // atom number
    float c[dim];
    search(x, c, tmp, tmp_int, &ano);
    uint64_t signs = 0;
    float cabs[dim];
    int nnz = 0;
    for (int i = 0; i < dim; i++) {
        cabs[i] = fabs(c[i]);
        if (c[i] != 0) {
            if (c[i] < 0)
                signs |= 1UL << nnz;
            nnz ++;
        }
    }
    const CodeSegment &cs = code_segments[ano];
    assert(nnz == cs.signbits);
    uint64_t code = cs.c0 + signs;
    code += cs.encode(cabs) << cs.signbits;
    return code;
}

uint64_t ZnSphereCodec::encode(const float *x) const
{
    return search_and_encode(x);
}


void ZnSphereCodec::decode(uint64_t code, float *c) const {
    int i0 = 0, i1 = natom;
    while (i0 + 1 < i1) {
        int imed = (i0 + i1) / 2;
        if (code_segments[imed].c0 <= code) i0 = imed;
        else i1 = imed;
    }
    const CodeSegment &cs = code_segments[i0];
    code -= cs.c0;
    uint64_t signs = code;
    code >>= cs.signbits;
    cs.decode(code, c);

    int nnz = 0;
    for (int i = 0; i < dim; i++) {
        if (c[i] != 0) {
            if (signs & (1UL << nnz))
                c[i] = -c[i];
            nnz ++;
        }
    }
}


/**************************************************************
 * ZnSphereCodecRec
 **************************************************************/

uint64_t ZnSphereCodecRec::get_nv(int ld, int r2a) const
{
    return all_nv[ld * (r2 + 1) + r2a];
}


uint64_t ZnSphereCodecRec::get_nv_cum(int ld, int r2t, int r2a) const
{
    return all_nv_cum[(ld * (r2 + 1) + r2t) * (r2 + 1) + r2a];
}

void ZnSphereCodecRec::set_nv_cum(int ld, int r2t, int r2a, uint64_t cum)
{
    all_nv_cum[(ld * (r2 + 1) + r2t) * (r2 + 1) + r2a] = cum;
}


ZnSphereCodecRec::ZnSphereCodecRec(int dim, int r2):
    VectorCodec(dim), r2(r2)
{
    log2_dim = 0;
    while (dim > (1 << log2_dim))
        log2_dim++;
    assert(dim == (1 << log2_dim) ||
           !"dimension must be a power of 2");

    all_nv.resize((log2_dim + 1) * (r2 + 1));
    all_nv_cum.resize((log2_dim + 1) * (r2 + 1) * (r2 + 1));

    for (int r2a = 0; r2a <= r2; r2a++) {
        int r = int(sqrt(r2a));
        if (r * r == r2a) {
            all_nv[r2a] = r == 0 ? 1 : 2;
        } else {
            all_nv[r2a] = 0;
        }
    }

    for (int ld = 1; ld <= log2_dim; ld++) {

        for (int r2sub = 0; r2sub <= r2; r2sub++) {
            uint64_t nv = 0;
            for (int r2a = 0; r2a <= r2sub; r2a++) {
                int r2b = r2sub - r2a;
                set_nv_cum(ld, r2sub, r2a, nv);
                nv += get_nv(ld - 1, r2a) * get_nv(ld - 1, r2b);
            }
            all_nv[ld * (r2 + 1) + r2sub] = nv;
        }
    }
    nv = get_nv(log2_dim, r2);

    uint64_t nvx = nv;
    code_size = 0;
    while (nvx > 0) {
        nvx >>= 8;
        code_size++;
    }

    int cache_level = std::min(3, log2_dim - 1);
    decode_cache_ld = 0;
    assert(cache_level <= log2_dim);
    decode_cache.resize((r2 + 1));

    for (int r2sub = 0; r2sub <= r2; r2sub++) {
        int ld = cache_level;
        uint64_t nvi = get_nv(ld, r2sub);
        std::vector<float> &cache = decode_cache[r2sub];
        int dimsub = (1 << cache_level);
        cache.resize (nvi * dimsub);
        float c[dim];
        uint64_t code0 = get_nv_cum(cache_level + 1, r2,
                                 r2 - r2sub);
        for (int i = 0; i < nvi; i++) {
            decode(i + code0, c);
            memcpy(&cache[i * dimsub], c + dim - dimsub,
                   dimsub * sizeof(*c));
        }
    }
    decode_cache_ld = cache_level;
}

uint64_t ZnSphereCodecRec::encode(const float *c) const
{
    return encode_centroid(c);
}



uint64_t ZnSphereCodecRec::encode_centroid(const float *c) const
{
    uint64_t codes[dim];
    int norm2s[dim];
    for(int i = 0; i < dim; i++) {
        if (c[i] == 0) {
            codes[i] = 0;
            norm2s[i] = 0;
        } else {
            int r2i = int(c[i] * c[i]);
            norm2s[i] = r2i;
            codes[i] = c[i] >= 0 ? 0 : 1;
        }
    }
    int dim2 = dim / 2;
    for(int ld = 1; ld <= log2_dim; ld++) {
        for (int i = 0; i < dim2; i++) {
            int r2a = norm2s[2 * i];
            int r2b = norm2s[2 * i + 1];

            uint64_t code_a = codes[2 * i];
            uint64_t code_b = codes[2 * i + 1];

            codes[i] =
                get_nv_cum(ld, r2a + r2b, r2a) +
                code_a * get_nv(ld - 1, r2b) +
                code_b;
            norm2s[i] = r2a + r2b;
        }
        dim2 /= 2;
    }
    return codes[0];
}



void ZnSphereCodecRec::decode(uint64_t code, float *c) const
{
    uint64_t codes[dim];
    int norm2s[dim];
    codes[0] = code;
    norm2s[0] = r2;

    int dim2 = 1;
    for(int ld = log2_dim; ld > decode_cache_ld; ld--) {
        for (int i = dim2 - 1; i >= 0; i--) {
            int r2sub = norm2s[i];
            int i0 = 0, i1 = r2sub + 1;
            uint64_t codei = codes[i];
            const uint64_t *cum =
                &all_nv_cum[(ld * (r2 + 1) + r2sub) * (r2 + 1)];
            while (i1 > i0 + 1) {
                int imed = (i0 + i1) / 2;
                if (cum[imed] <= codei)
                    i0 = imed;
                else
                    i1 = imed;
            }
            int r2a = i0, r2b = r2sub - i0;
            codei -= cum[r2a];
            norm2s[2 * i] = r2a;
            norm2s[2 * i + 1] = r2b;

            uint64_t code_a = codei / get_nv(ld - 1, r2b);
            uint64_t code_b = codei % get_nv(ld - 1, r2b);

            codes[2 * i] = code_a;
            codes[2 * i + 1] = code_b;

        }
        dim2 *= 2;
    }

    if (decode_cache_ld == 0) {
        for(int i = 0; i < dim; i++) {
            if (norm2s[i] == 0) {
                c[i] = 0;
            } else {
                float r = sqrt(norm2s[i]);
                assert(r * r == norm2s[i]);
                c[i] = codes[i] == 0 ? r : -r;
            }
        }
    } else {
        int subdim = 1 << decode_cache_ld;
        assert ((dim2 * subdim) == dim);

        for(int i = 0; i < dim2; i++) {

            const std::vector<float> & cache =
                decode_cache[norm2s[i]];
            assert(codes[i] < cache.size());
            memcpy(c + i * subdim,
                   &cache[codes[i] * subdim],
                   sizeof(*c)* subdim);
        }
    }
}
