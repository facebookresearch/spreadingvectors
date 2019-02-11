/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <vector>
#include <stdint.h>
#include <cstdio>


inline float sqr(float x) {
    return x * x;
}

inline int popcount64(uint64_t x) {
    return __builtin_popcountl(x);
}

typedef std::vector<float> point_list_t;

/** compute combinations of n integer values <= v that sum up to total
 * (squared).
 *
 * if ret is the returned point_list_t, the number of vectors is
 * ret.size() / n
 */
point_list_t sum_of_sq (float total, int v, int n, float add=0);


inline float dotprod(int n, const float *a, const float *b) {
    float accu = 0;
    for(int i = 0; i < n; i++)
        accu += a[i] * b[i];
    return accu;
}

struct Comb {
    std::vector<uint64_t> tab; // Pascal's triangle
    int nmax;
    Comb(int nmax);
    uint64_t operator()(int n, int p) const {
        if (p > n) return 0;
        return tab[n * nmax + p];
    }
};

// initialized with nmax=100
extern Comb comb;

struct Repeat {
    float val;
    int n;
};


/** Repeats: used to encode a vector that has n occurrences of
 *  val. Encodes the signs and permutation of the vector. Useful for
 *  atoms.
 */
struct Repeats {
    int dim;
    std::vector<Repeat> repeats;

    // initialize from a template of the atom.
    Repeats(int dim = 0, const float *c = nullptr);

    // count number of possible codes for this atom
    long count() const;

    long encode(const float *c) const;

    void decode(uint64_t code, float *c) const;
};


// optimized inner product
float  fvec_inner_product (const float * x,
                           const float * y,
                           size_t d);
