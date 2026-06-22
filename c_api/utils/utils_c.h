/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_UTILS_C_H
#define FAISS_UTILS_C_H

#include <stdint.h>
#include <stdlib.h>
#include "../faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* faiss_get_version();

/** Convert real-valued vectors to binary vectors.
 *
 * Bits are packed LSB-first within each output byte: bit j corresponds to
 * x_in[8 * i + j] for output byte i. A bit is set to 1 if the corresponding
 * input value is greater than 0, and 0 otherwise.
 *
 * @param n      number of vectors
 * @param d      dimension of each input vector (must be a multiple of 8)
 * @param x_in   input float vectors (float table of size n * d)
 * @param x_out  output binary vectors (uint8_t table of size n * (d / 8))
 */
void faiss_real_to_binary(
        size_t n,
        size_t d,
        const float* x_in,
        uint8_t* x_out);

/** Merge k sorted result lists from nshard shards into a single top-k list.
 *
 * Both all_distances and all_labels are row-major with layout (nshard, n, k).
 * Outputs distances and labels have layout (n, k), sorted best-first.
 * Each per-shard list must already be sorted (best result at index 0).
 *
 * @param n              number of query vectors
 * @param k              number of results per query
 * @param nshard         number of shards to merge
 * @param keep_min       1 for L2 (smallest distance wins), 0 for IP (largest)
 * @param all_distances  input distances, size nshard * n * k
 * @param all_labels     input labels,    size nshard * n * k
 * @param distances      output distances, size n * k
 * @param labels         output labels,    size n * k
 */
int faiss_merge_knn_results(
        size_t n,
        size_t k,
        int nshard,
        int keep_min,
        const float* all_distances,
        const idx_t* all_labels,
        float* distances,
        idx_t* labels);

#ifdef __cplusplus
}
#endif

#endif
