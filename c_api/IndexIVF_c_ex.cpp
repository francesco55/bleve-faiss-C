/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2023-present Facebook. All Rights Reserved.
// -*- c++ -*-
#include "IndexIVF_c_ex.h"
#include <faiss/IndexIVF.h>
#include <memory>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/clone_index.h>
#include <faiss/impl/FaissAssert.h>
#include "macros_impl.h"

using faiss::IndexIVF;
using faiss::IndexIVFRaBitQ;
using faiss::SearchParameters;
using faiss::SearchParametersIVF;

int faiss_IndexIVF_set_direct_map(FaissIndexIVF* index, int direct_map_type) {
    try {
        reinterpret_cast<IndexIVF*>(index)->set_direct_map_type(
                static_cast<faiss::DirectMap::Type>(direct_map_type));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_list_vector_count(
        const FaissIndexIVF* index,
        idx_t* list_counts,
        size_t list_counts_size,
        const FaissSearchParametersIVF* params) {
    try {
        reinterpret_cast<const IndexIVF*>(index)->ivf_list_vector_count(
                list_counts,
                list_counts_size,
                reinterpret_cast<const faiss::SearchParameters*>(params));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_search_closest_eligible_centroids(
        const FaissIndexIVF* index,
        idx_t n,
        const float* query,
        idx_t k,
        float* centroid_distances,
        idx_t* centroid_ids,
        const FaissSearchParameters* params) {
    try {
        reinterpret_cast<const IndexIVF*>(index)->quantizer->search(
                n,
                query,
                k,
                centroid_distances,
                centroid_ids,
                reinterpret_cast<const faiss::SearchParameters*>(params));
    }
    CATCH_AND_HANDLE
}

int faiss_Set_quantizers(FaissIndex* target, FaissIndex* source) {
    try {
        auto* tgt = reinterpret_cast<faiss::Index*>(target);
        auto* src = reinterpret_cast<faiss::Index*>(source);

        assert(tgt && src);

        // -------- IndexIVFScalarQuantizer --------
        if (auto* tgt_ivfsq = dynamic_cast<faiss::IndexIVFScalarQuantizer*>(tgt)) {
            auto* src_ivfsq = dynamic_cast<faiss::IndexIVFScalarQuantizer*>(src);
            assert(src_ivfsq);

            tgt_ivfsq->quantizer = src_ivfsq->quantizer;
            tgt_ivfsq->is_trained = true;
            tgt_ivfsq->sq = src_ivfsq->sq;
            return 0;
        }

        // --------- IndexSQ ---------
        if (auto* tgt_sq = dynamic_cast<faiss::IndexScalarQuantizer*>(tgt)) {
            auto* src_sq = dynamic_cast<faiss::IndexScalarQuantizer*>(src);
            assert(src_sq);

            tgt_sq->is_trained = true;
            tgt_sq->sq = src_sq->sq;
            return 0;
        }

        // -------- IndexIVFRaBitQ --------
        if (auto* tgt_rabitq = dynamic_cast<faiss::IndexIVFRaBitQ*>(tgt)) {
            auto* src_rabitq = dynamic_cast<faiss::IndexIVFRaBitQ*>(src);
            assert(src_rabitq);

            tgt_rabitq->quantizer = src_rabitq->quantizer;
            tgt_rabitq->is_trained = true;
            tgt_rabitq->rabitq = src_rabitq->rabitq;
            return 0;
        }

        // Unsupported type
        return -1;
    }
    CATCH_AND_HANDLE
}

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
        const FaissSearchParametersIVF* params) {
    try {
        reinterpret_cast<const IndexIVF*>(index)->search_preassigned(
                n,
                x,
                k,
                assign,
                centroid_dis,
                distances,
                labels,
                store_pairs,
                reinterpret_cast<const faiss::SearchParametersIVF*>(params));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_compute_distance_to_codes_for_list(
        FaissIndexIVF* index,
        idx_t list_no,
        const float* x,
        idx_t n,
        const uint8_t* codes,
        float* dists,
        float* dist_table) {
    try {
        reinterpret_cast<IndexIVF*>(index)->compute_distance_to_codes_for_list(
                list_no, x, n, codes, dists, dist_table);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_get_centroids_and_cardinality(
        const FaissIndexIVF* index,
        float* centroid_vectors,
        size_t* cardinalities,
        idx_t* centroid_ids) {
    try {
        reinterpret_cast<const IndexIVF*>(index)->get_centroids_and_cardinality(
            centroid_vectors, cardinalities, centroid_ids);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_compute_distance_table(
        FaissIndexIVF* index,
        const float* x,
        float* dist_table) {
    try {
        reinterpret_cast<IndexIVF*>(index)->compute_distance_table(
                x, dist_table);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_has_RaBitQ(FaissIndex* index) {
    try {
        faiss::Index* idx = reinterpret_cast<faiss::Index*>(index);

        faiss::IndexIVFRaBitQ* ivf_rq =
            dynamic_cast<faiss::IndexIVFRaBitQ*>(idx);

        if (ivf_rq) {
            return 0;
        }

        return -1;
    }
    CATCH_AND_HANDLE
}

int faiss_SearchParametersRaBitQ_new_with(
        FaissSearchParametersIVF** p_sp,
        FaissIDSelector* sel,
        size_t nprobe,
        size_t max_codes) {
    try {
        faiss::IVFRaBitQSearchParameters* rqsp = new faiss::IVFRaBitQSearchParameters;
        rqsp->centered = true;
        rqsp->qb = 4;
        rqsp->sel = reinterpret_cast<faiss::IDSelector*>(sel);
        rqsp->nprobe = nprobe;
        rqsp->max_codes = max_codes;

        *p_sp = reinterpret_cast<FaissSearchParametersIVF*>(rqsp);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFRaBitQ_compute_distance_with_precomputed(
        FaissIndexIVF* index,
        idx_t list_no,
        const float* x,
        idx_t n,
        const uint8_t* codes,
        float* dists,
        uint8_t* query_bp,
        size_t* query_bp_size) {
    try {
        auto* rabitq_index = dynamic_cast<IndexIVFRaBitQ*>(
                reinterpret_cast<IndexIVF*>(index));
        FAISS_THROW_IF_NOT_MSG(
                rabitq_index,
                "index is not an IndexIVFRaBitQ instance");
        rabitq_index->compute_distance_to_codes_with_precomputed(
                list_no, x, n, codes, dists,
                query_bp, query_bp_size);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFRaBitQ_query_bitplanes_size(
        FaissIndexIVF* index,
        size_t* size) {
    try {
        auto* rabitq_index = dynamic_cast<IndexIVFRaBitQ*>(
                reinterpret_cast<IndexIVF*>(index));
        FAISS_THROW_IF_NOT_MSG(
                rabitq_index,
                "index is not an IndexIVFRaBitQ instance");
        *size = rabitq_index->query_bitplanes_size();
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_init_partition_map(FaissIndexIVF* index, int my_worker_id) {
    try {
        auto* ivf = reinterpret_cast<IndexIVF*>(index);
        ivf->partition_map = std::make_shared<faiss::IVFPartitionMap>(
                ivf->nlist, my_worker_id);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_set_list_worker(
        FaissIndexIVF* index,
        size_t list_no,
        int worker_id) {
    try {
        auto* ivf = reinterpret_cast<IndexIVF*>(index);
        FAISS_THROW_IF_NOT_MSG(
                ivf->partition_map != nullptr,
                "partition map not initialized; call faiss_IndexIVF_init_partition_map first");
        FAISS_THROW_IF_NOT_FMT(
                list_no < ivf->nlist,
                "list_no %zu out of range [0, %zu)",
                list_no,
                ivf->nlist);
        ivf->partition_map->list_to_worker[list_no] = worker_id;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_get_list_worker(
        const FaissIndexIVF* index,
        size_t list_no,
        int* out_worker_id) {
    try {
        const auto* ivf = reinterpret_cast<const IndexIVF*>(index);
        FAISS_THROW_IF_NOT_MSG(
                ivf->partition_map != nullptr,
                "partition map not initialized");
        FAISS_THROW_IF_NOT_FMT(
                list_no < ivf->nlist,
                "list_no %zu out of range [0, %zu)",
                list_no,
                ivf->nlist);
        *out_worker_id = ivf->partition_map->owner(list_no);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_has_partition_map(const FaissIndexIVF* index) {
    const auto* ivf = reinterpret_cast<const IndexIVF*>(index);
    return ivf->partition_map != nullptr ? 1 : 0;
}

int faiss_IndexIVF_init_partition_map_with_owners(
        FaissIndexIVF* index,
        int my_worker_id,
        const int* list_to_worker,
        size_t n) {
    try {
        auto* ivf = reinterpret_cast<IndexIVF*>(index);
        FAISS_THROW_IF_NOT_FMT(
                n == ivf->nlist,
                "list_to_worker length %zu does not match nlist %zu",
                n,
                ivf->nlist);
        ivf->partition_map =
                std::make_shared<faiss::IVFPartitionMap>(n, my_worker_id);
        std::copy(
                list_to_worker,
                list_to_worker + n,
                ivf->partition_map->list_to_worker.begin());
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_set_list_workers(
        FaissIndexIVF* index,
        const idx_t* list_nos,
        const int* worker_ids,
        size_t n) {
    try {
        auto* ivf = reinterpret_cast<IndexIVF*>(index);
        FAISS_THROW_IF_NOT_MSG(
                ivf->partition_map != nullptr,
                "partition map not initialized; call faiss_IndexIVF_init_partition_map first");
        for (size_t i = 0; i < n; i++) {
            FAISS_THROW_IF_NOT_FMT(
                    (size_t)list_nos[i] < ivf->nlist,
                    "list_no %zu out of range [0, %zu)",
                    (size_t)list_nos[i],
                    ivf->nlist);
            ivf->partition_map->list_to_worker[list_nos[i]] = worker_ids[i];
        }
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_get_list_workers(
        const FaissIndexIVF* index,
        const idx_t* list_nos,
        size_t n,
        int* out_worker_ids) {
    try {
        const auto* ivf = reinterpret_cast<const IndexIVF*>(index);
        FAISS_THROW_IF_NOT_MSG(
                ivf->partition_map != nullptr,
                "partition map not initialized");
        for (size_t i = 0; i < n; i++) {
            FAISS_THROW_IF_NOT_FMT(
                    (size_t)list_nos[i] < ivf->nlist,
                    "list_no %zu out of range [0, %zu)",
                    (size_t)list_nos[i],
                    ivf->nlist);
            out_worker_ids[i] = ivf->partition_map->owner(list_nos[i]);
        }
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_copy_lists_to(
        const FaissIndexIVF* src,
        FaissIndexIVF* dst,
        const idx_t* list_nos,
        size_t n_lists) {
    try {
        reinterpret_cast<const IndexIVF*>(src)->copy_lists_to(
                *reinterpret_cast<IndexIVF*>(dst), list_nos, n_lists);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_search_closest_centroids_with_workers(
        const FaissIndexIVF* index,
        idx_t n,
        const float* x,
        idx_t nprobe,
        int* out_worker_ids,
        idx_t* out_centroid_ids,
        float* out_distances) {
    try {
        const auto* ivf = reinterpret_cast<const IndexIVF*>(index);
        FAISS_THROW_IF_NOT_MSG(
                ivf->partition_map != nullptr,
                "partition map not initialized; call faiss_IndexIVF_init_partition_map first");
        FAISS_THROW_IF_NOT_MSG(nprobe > 0, "nprobe must be > 0");

        ivf->quantizer->search(n, x, nprobe, out_distances, out_centroid_ids);

        const idx_t total = n * nprobe;
        for (idx_t i = 0; i < total; i++) {
            idx_t cid = out_centroid_ids[i];
            out_worker_ids[i] = (cid >= 0) ? ivf->partition_map->owner(cid) : -1;
        }
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_train_and_add(
        FaissIndexIVF* index,
        idx_t n,
        const float* x) {
    try {
        auto* ivf = reinterpret_cast<IndexIVF*>(index);
        ivf->train(n, x);
        ivf->add(n, x);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_train_and_add_with_ids(
        FaissIndexIVF* index,
        idx_t n,
        const float* x,
        const idx_t* xids) {
    try {
        auto* ivf = reinterpret_cast<IndexIVF*>(index);
        ivf->train(n, x);
        ivf->add_with_ids(n, x, xids);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_set_quantizer_centroids(
        FaissIndexIVF* index,
        const float* centroids,
        size_t nlist,
        size_t d) {
    try {
        auto* ivf = reinterpret_cast<IndexIVF*>(index);
        FAISS_THROW_IF_NOT_MSG(
                ivf->nlist == nlist,
                "set_quantizer_centroids: nlist does not match index->nlist");
        FAISS_THROW_IF_NOT_MSG(
                static_cast<size_t>(ivf->d) == d,
                "set_quantizer_centroids: d does not match index->d");
        // Replace the coarse-quantizer contents with the given centroids in order
        // (inverted list i == centroids[i]); no k-means. The quantizer for an
        // IVF*,Flat index is a flat index, which is always trained, so adding the
        // centroids is sufficient to mark the whole index trained.
        ivf->quantizer->reset();
        ivf->quantizer->add(static_cast<idx_t>(nlist), centroids);
        ivf->is_trained = true;
    }
    CATCH_AND_HANDLE
}
