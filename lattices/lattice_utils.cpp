/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "lattice_utils.h"

#include <cassert>

#include <unordered_set>

#include <immintrin.h>


// compute combinations of n integer values <= v that sum up to total (squared)
point_list_t sum_of_sq (float total, int v, int n, float add) {
    if (total < 0) {
        return point_list_t();
    } else if (n == 1) {
        while (sqr(v + add) > total) v--;
        if (sqr(v + add) == total) {
            return point_list_t(1, v + add);
        } else {
            return point_list_t();
        }
    } else {
        point_list_t res;
        while (v >= 0) {
            point_list_t sub_points = sum_of_sq (total - sqr(v + add), v, n - 1, add);
            for (size_t i = 0; i < sub_points.size(); i += n - 1) {
                res.push_back (v + add);
                for (int j = 0; j < n - 1; j++)
                    res.push_back(sub_points[i + j]);
            }
            v--;
        }
        return res;
    }
}



Comb::Comb(int nmax):nmax(nmax) {
    tab.resize(nmax * nmax, 0);
    tab[0] = 1;
    for(int i = 1; i < nmax; i++) {
        tab[i * nmax] = 1;
        for(int j = 1; j <= i; j++) {
            tab[i * nmax + j] =
                tab[(i - 1) * nmax + j] +
                tab[(i - 1) * nmax + (j - 1)];
        }

    }
}

Comb comb(100);




Repeats::Repeats (int dim, const float *c): dim(dim)
{
    for(int i = 0; i < dim; i++) {
        int j = 0;
        for(;;) {
            if (j == repeats.size()) {
                repeats.push_back(Repeat{c[i], 1});
                break;
            }
            if (repeats[j].val == c[i]) {
                repeats[j].n++;
                break;
            }
            j++;
        }
    }
}


long Repeats::count () const
{
    long accu = 1;
    int remain = dim;
    for (int i = 0; i < repeats.size(); i++) {
        accu *= comb(remain, repeats[i].n);
        remain -= repeats[i].n;
    }
    return accu;
}

// optimized version for < 64 bits
static long repeats_encode_64 (
     const std::vector<Repeat> & repeats,
     int dim, const float *c)
{
    uint64_t coded = 0;
    int nfree = dim;
    uint64_t code = 0, shift = 1;
    for (auto r = repeats.begin(); r != repeats.end(); ++r) {
        int rank = 0, occ = 0;
        uint64_t code_comb = 0;
        uint64_t tosee = ~coded;
        for(;;) {
            // directly jump to next available slot.
            int i = __builtin_ctzl(tosee);
            tosee &= ~(1UL << i) ;
            if (c[i] == r->val) {
                code_comb += comb(rank, occ + 1);
                occ++;
                coded |= 1UL << i;
                if (occ == r->n) break;
            }
            rank++;
        }
        uint64_t max_comb = comb(nfree, r->n);
        code += shift * code_comb;
        shift *= max_comb;
        nfree -= r->n;
    }
    return code;
}



// version with a bool vector that works for > 64 dim
long Repeats::encode(const float *c) const
{
    if (dim < 64)
        return repeats_encode_64 (repeats, dim, c);
    std::vector<bool> coded(dim, false);
    int nfree = dim;
    uint64_t code = 0, shift = 1;
    for (auto r = repeats.begin(); r != repeats.end(); ++r) {
        int rank = 0, occ = 0;
        uint64_t code_comb = 0;
        for (int i = 0; i < dim; i++) {
            if (!coded[i]) {
                if (c[i] == r->val) {
                    code_comb += comb(rank, occ + 1);
                    occ++;
                    coded[i] = true;
                    if (occ == r->n) break;
                }
                rank++;
            }
        }
        uint64_t max_comb = comb(nfree, r->n);
        code += shift * code_comb;
        shift *= max_comb;
        nfree -= r->n;
    }
    return code;
}





static int decode_comb_1 (uint64_t *n, int k1, int r) {
    while (comb(r, k1) > *n)
        r--;
    *n -= comb(r, k1);
    return r;
}


static void repeats_decode_64(
     const std::vector<Repeat> & repeats,
     int dim, uint64_t code, float *c)
{
    uint64_t decoded = 0;
    int nfree = dim;
    for (auto r = repeats.begin(); r != repeats.end(); ++r) {
        uint64_t max_comb = comb(nfree, r->n);
        uint64_t code_comb = code % max_comb;
        code /= max_comb;

        int occ = 0;
        int rank = nfree;
        int next_rank = decode_comb_1 (&code_comb, r->n, rank);
        uint64_t tosee = ((1UL << dim) - 1) ^ decoded;
        for(;;) {
            int i = 63 - __builtin_clzl(tosee);
            tosee &= ~(1UL << i);
            rank--;
            if (rank == next_rank) {
                decoded |= 1UL << i;
                c[i] = r->val;
                occ++;
                if (occ == r->n) break;
                next_rank = decode_comb_1 (
                   &code_comb, r->n - occ, next_rank);
            }
        }
        nfree -= r->n;
    }

}



void Repeats::decode(uint64_t code, float *c) const
{
    if (dim < 64) {
        repeats_decode_64 (repeats, dim, code, c);
        return;
    }

    std::vector<bool> decoded(dim, false);
    int nfree = dim;
    for (auto r = repeats.begin(); r != repeats.end(); ++r) {
        uint64_t max_comb = comb(nfree, r->n);
        uint64_t code_comb = code % max_comb;
        code /= max_comb;

        int occ = 0;
        int rank = nfree;
        int next_rank = decode_comb_1 (&code_comb, r->n, rank);
        for (int i = dim - 1; i >= 0; i--) {
            if (!decoded[i]) {
                rank--;
                if (rank == next_rank) {
                    decoded[i] = true;
                    c[i] = r->val;
                    occ++;
                    if (occ == r->n) break;
                    next_rank = decode_comb_1 (
                         &code_comb, r->n - occ, next_rank);
                }
            }
        }
        nfree -= r->n;
    }

}



// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read (int d, const float *x)
{
    assert (0 <= d && d < 4);
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
    switch (d) {
      case 3:
        buf[2] = x[2];
      case 2:
        buf[1] = x[1];
      case 1:
        buf[0] = x[0];
    }
    return _mm_load_ps (buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}

float fvec_inner_product (const float * x,
                          const float * y,
                          size_t d)
{
    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        msum1 = _mm256_add_ps (msum1, _mm256_mul_ps (mx, my));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 +=       _mm256_extractf128_ps(msum1, 0);

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        msum2 = _mm_add_ps (msum2, _mm_mul_ps (mx, my));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        msum2 = _mm_add_ps (msum2, _mm_mul_ps (mx, my));
    }

    msum2 = _mm_hadd_ps (msum2, msum2);
    msum2 = _mm_hadd_ps (msum2, msum2);
    return  _mm_cvtss_f32 (msum2);
}

float fvec_L2sqr (const float * x,
                  const float * y,
                  size_t d)
{
    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        const __m256 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 +=       _mm256_extractf128_ps(msum1, 0);

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        const __m128 a_m_b1 = mx - my;
        msum2 += a_m_b1 * a_m_b1;
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        __m128 a_m_b1 = mx - my;
        msum2 += a_m_b1 * a_m_b1;
    }

    msum2 = _mm_hadd_ps (msum2, msum2);
    msum2 = _mm_hadd_ps (msum2, msum2);
    return  _mm_cvtss_f32 (msum2);
}
