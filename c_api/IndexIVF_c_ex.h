/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2023-present Facebook. All Rights Reserved.
// -*- c -*-

#ifndef FAISS_INDEX_IVF_EX_C_H
#define FAISS_INDEX_IVF_EX_C_H

#include "Clustering_c.h"
#include "IndexIVF_c.h"
#include "Index_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
    Set the direct map type for the IVF index.

    @param index            - Pointer to the Faiss IVF index
    @param direct_map_type  - Type of direct map to set (faiss::DirectMap::Type)
*/
int faiss_IndexIVF_set_direct_map(FaissIndexIVF* index, int direct_map_type);

/*
    Count vectors per IVF list (cluster) for a given selector.

    This function iterates over the vectors selected by the provided
    search parameters and increments a counter for each IVF inverted
    list (cluster) they belong to. The result is a per-list vector
    count for the IVF index.

    @param index             - Pointer to the Faiss IVF index
    @param list_counts       - Output array of size index->nlist. Must be
                               initialized to zero by the caller. On return,
                               list_counts[i] contains the number of selected
                               vectors assigned to IVF list i.
    @param list_counts_size  - Size of list_counts array (must equal index->nlist)
    @param params            - IVF search parameters containing the selector
                               that defines which vectors are included
*/
int faiss_IndexIVF_list_vector_count(
        const FaissIndexIVF* index,
        idx_t* list_counts,
        size_t list_counts_size,
        const FaissSearchParametersIVF* params);

/*
    Return 'k' centroids in the index closest to the query vector.

    @param n: number of queries.
    @param query: query vector.
    @param k: count of closest number of vectors.
    @param centroid_distances: output distances, size n * k.
    @param centroid_ids: output centroid IDs, size n * k.
*/
int faiss_IndexIVF_search_closest_eligible_centroids(
        const FaissIndexIVF* index,
        idx_t n,
        const float* query,
        idx_t k,
        float* centroid_distances,
        idx_t* centroid_ids,
        const FaissSearchParameters* params);

/*
    Search the clusters whose IDs are in 'assign' and
    return the 'k' nearest neighbours from among them.

    @param n: number of queries.
    @param x: query vector, size n * d.
    @param k: count of nearest neighbours to be returned for each query.
    @param centroid_ids: output centroid IDs, size n * k.
    @param distance: output distances, size n * k
    @param labels: output labels, size n * k
*/
int faiss_IndexIVF_search_preassigned_with_params(
        const FaissIndexIVF* index,
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        int store_pairs,
        const FaissSearchParametersIVF* params);

/*
    Given a query vector x, compute distance to provided codes
    for the input list_no. This is a special purpose method
    to be used as a flat distance computer for an inverted
    list where codes are provided externally. This allows to
    use the quantizer independently while computing distance
    for the quantized codes.

    @param list_no list number for inverted list
    @param x - input query vector
    @param n - number of codes
    @param codes - input codes
    @param dists - output computed distances
    @param dist_table - input precomputed distance table for PQ
*/

int faiss_IndexIVF_compute_distance_to_codes_for_list(
        FaissIndexIVF* index,
        idx_t list_no,
        const float* x,
        idx_t n,
        const uint8_t* codes,
        float* dists,
        float* dist_table);

/*
    Compute distance to codes with optional precomputed query
    bitplanes.

    query_bp: caller-allocated buffer of
    faiss_IndexIVFRaBitQ_query_bitplanes_size() bytes.
    query_bp_size: in/out. On entry: 0 or mismatched header =
    (re)compute and fill query_bp; matching size + valid header =
    reuse existing precomputed data. On return: set to actual
    bytes written.

    @param index          - the IVF index (must be IndexIVFRaBitQ)
    @param list_no        - list number for inverted list
    @param x              - input query vector
    @param n              - number of codes
    @param codes          - input codes
    @param dists          - output computed distances
    @param query_bp       - precomputed query bitplanes buffer
    @param query_bp_size  - in/out: 0 to compute, >0 to reuse
*/
int faiss_IndexIVFRaBitQ_compute_distance_with_precomputed(
        FaissIndexIVF* index,
        idx_t list_no,
        const float* x,
        idx_t n,
        const uint8_t* codes,
        float* dists,
        uint8_t* query_bp,
        size_t* query_bp_size);

/*
    Get the byte size needed for precomputed query bitplanes buffer.

    @param index  - the IVF index (must be IndexIVFRaBitQ)
    @param size   - output: required buffer size in bytes
*/
int faiss_IndexIVFRaBitQ_query_bitplanes_size(
        FaissIndexIVF* index,
        size_t* size);

/*
    Get centroid information and cardinality for all centroids in an IVF index.

    @param index: the IVF index
    @param centroid_vectors: output array for centroid vectors, size nlist * d
    @param cardinalities: output array for cardinalities, size nlist
    @param centroid_ids: output array for centroid IDs, size nlist (optional, can be nullptr)
*/
int faiss_IndexIVF_get_centroids_and_cardinality(
        const FaissIndexIVF* index,
        float* centroid_vectors,
        size_t* cardinalities,
        idx_t* centroid_ids);

/*
    Given a query vector x, compute distance table and
    return to the caller.

    @param x - input query vector
    @param dist_table - output precomputed distance table for PQ

*/

int faiss_IndexIVF_compute_distance_table(
        FaissIndexIVF* index,
        const float* x,
        float* dist_table);

/*
        Given a source index, clone the 
          coarse quantizer having the centroid layout information,
          quantizer having details specific to how the codes themselves were encoded (PQ, SQ etc.)
        and set them in the target index

        @param target - the target index to set the quantizers
        @param source - the source index to clone the quantizers from
*/

int faiss_IndexIVF_has_RaBitQ(
        FaissIndex* index);

int faiss_SearchParametersRaBitQ_new_with(
        FaissSearchParametersIVF** p_sp,
        FaissIDSelector* sel,
        size_t nprobe,
        size_t max_codes);

int faiss_Set_quantizers(
        FaissIndex* target,
        FaissIndex* source);

/*
    Initialize a partition map on the index.

    Allocates an nlist-sized map with all entries set to -1 (unassigned)
    and records my_worker_id as the ID of the current node.  Must be
    called before faiss_IndexIVF_set_list_worker.

    @param index         - Pointer to the Faiss IVF index
    @param my_worker_id  - Worker ID of the node holding this index
*/
int faiss_IndexIVF_init_partition_map(FaissIndexIVF* index, int my_worker_id);

/*
    Set the owning worker for a single inverted list (== centroid).

    @param index      - Pointer to the Faiss IVF index
    @param list_no    - Inverted list number (== centroid number)
    @param worker_id  - Worker ID that owns this list
*/
int faiss_IndexIVF_set_list_worker(
        FaissIndexIVF* index,
        size_t list_no,
        int worker_id);

/*
    Batch-write the owning worker ID for an arbitrary set of inverted lists.

    @param index       - Pointer to the Faiss IVF index
    @param list_nos    - Inverted list numbers (== centroid numbers) to assign, size n
    @param worker_ids  - Worker ID for each list, size n
    @param n           - Number of entries in list_nos / worker_ids
*/
int faiss_IndexIVF_set_list_workers(
        FaissIndexIVF* index,
        const idx_t* list_nos,
        const int* worker_ids,
        size_t n);

/*
    Get the owning worker for a single inverted list (== centroid).

    @param index          - Pointer to the Faiss IVF index
    @param list_no        - Inverted list number (== centroid number)
    @param out_worker_id  - Output: worker ID that owns this list, or -1 if unset
*/
int faiss_IndexIVF_get_list_worker(
        const FaissIndexIVF* index,
        size_t list_no,
        int* out_worker_id);

/*
    Returns 1 if the index has a partition map set, 0 otherwise.

    @param index  - Pointer to the Faiss IVF index
*/
int faiss_IndexIVF_has_partition_map(const FaissIndexIVF* index);

/*
    Initialize the partition map and bulk-write all list→worker assignments
    in a single call.  Replaces any existing partition map.

    @param index          - Pointer to the Faiss IVF index
    @param my_worker_id   - Worker ID of the node holding this index
    @param list_to_worker - Array of worker IDs, one per list; length must equal nlist
    @param n              - Length of list_to_worker (must equal nlist)
*/
int faiss_IndexIVF_init_partition_map_with_owners(
        FaissIndexIVF* index,
        int my_worker_id,
        const int* list_to_worker,
        size_t n);

/*
    Batch-read the owning worker ID for an arbitrary set of inverted lists.

    @param index          - Pointer to the Faiss IVF index
    @param list_nos       - Inverted list numbers (== centroid numbers) to query, size n
    @param n              - Number of entries in list_nos / out_worker_ids
    @param out_worker_ids - Output: worker ID for each list, size n
*/
int faiss_IndexIVF_get_list_workers(
        const FaissIndexIVF* index,
        const idx_t* list_nos,
        size_t n,
        int* out_worker_ids);

/*
    Copy an arbitrary set of inverted lists from src to dst.

    Both indexes must have the same nlist and code_size.
    dst->ntotal is updated to reflect the vectors added.
    Lists that are empty in the source are silently skipped.

    @param src       - Source index
    @param dst       - Destination index (same nlist / code_size)
    @param list_nos  - List numbers (== centroid numbers) to copy, size n_lists
    @param n_lists   - Number of lists to copy
*/
int faiss_IndexIVF_copy_lists_to(
        const FaissIndexIVF* src,
        FaissIndexIVF* dst,
        const idx_t* list_nos,
        size_t n_lists);

#ifdef __cplusplus
}
#endif

#endif
