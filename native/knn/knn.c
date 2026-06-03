#include <erl_nif.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <stdlib.h>

#define DIMS 8
#define N_CLUSTERS 4096

typedef struct {
    float dist;
    unsigned char label;
} knn_entry;

typedef struct {
    float dist;
    int idx;
} cluster_dist;

static inline float dist_sq_avx_8d(const float* q, const float* v) {
    __m256 qv = _mm256_loadu_ps(q);
    __m256 vv = _mm256_loadu_ps(v);
    __m256 diff = _mm256_sub_ps(qv, vv);
    __m256 sq = _mm256_mul_ps(diff, diff);

    __m128 low = _mm256_castps256_ps128(sq);
    __m128 high = _mm256_extractf128_ps(sq, 1);
    __m128 sum128 = _mm_add_ps(low, high);

    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);

    return _mm_cvtss_f32(sums);
}

static int compare_cluster_dist(const void* a, const void* b) {
    float da = ((const cluster_dist*)a)->dist;
    float db = ((const cluster_dist*)b)->dist;
    return (da > db) - (da < db);
}

static void insert_top_k(knn_entry* top_k, int* top_k_len, int* max_idx, float* max_dist, float dist, unsigned char label, int k) {
    if (*top_k_len < k) {
        top_k[*top_k_len].dist = dist;
        top_k[*top_k_len].label = label;
        if (dist > *max_dist) *max_dist = dist;
        (*top_k_len)++;
        // Recompute max
        *max_dist = top_k[0].dist;
        *max_idx = 0;
        for (int j = 1; j < *top_k_len; j++) {
            if (top_k[j].dist > *max_dist) {
                *max_dist = top_k[j].dist;
                *max_idx = j;
            }
        }
    } else if (dist < *max_dist) {
        top_k[*max_idx].dist = dist;
        top_k[*max_idx].label = label;
        // Recompute max
        *max_dist = top_k[0].dist;
        *max_idx = 0;
        for (int j = 1; j < k; j++) {
            if (top_k[j].dist > *max_dist) {
                *max_dist = top_k[j].dist;
                *max_idx = j;
            }
        }
    }
}

static ERL_NIF_TERM knn_search_ivf(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary vectors_bin, labels_bin, centroids_bin, bucket_starts_bin, query_bin;
    unsigned int k, nprobe, n_clusters;

    if (!enif_inspect_binary(env, argv[0], &vectors_bin)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[1], &labels_bin)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &centroids_bin)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[3], &bucket_starts_bin)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[4], &query_bin)) return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[5], &k)) return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[6], &nprobe)) return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[7], &n_clusters)) return enif_make_badarg(env);

    float* query = (float*)query_bin.data;
    float* vectors = (float*)vectors_bin.data;
    unsigned char* labels = labels_bin.data;
    float* centroids = (float*)centroids_bin.data;
    int* bucket_starts = (int*)bucket_starts_bin.data;

    // 1. Compute distances to all centroids
    cluster_dist* cluster_dists = (cluster_dist*)enif_alloc(n_clusters * sizeof(cluster_dist));
    for (int c = 0; c < n_clusters; c++) {
        float* centroid = centroids + c * DIMS;
        float d2 = dist_sq_avx_8d(query, centroid);
        cluster_dists[c].dist = d2;
        cluster_dists[c].idx = c;
    }

    // 2. Partial sort to get top nprobe clusters
    int actual_nprobe = nprobe < n_clusters ? nprobe : n_clusters;
    if (actual_nprobe < 100) {
        for (int i = 0; i < actual_nprobe; i++) {
            int min_idx = i;
            for (int j = i + 1; j < n_clusters; j++) {
                if (cluster_dists[j].dist < cluster_dists[min_idx].dist) {
                    min_idx = j;
                }
            }
            if (min_idx != i) {
                cluster_dist tmp = cluster_dists[i];
                cluster_dists[i] = cluster_dists[min_idx];
                cluster_dists[min_idx] = tmp;
            }
        }
    } else {
        qsort(cluster_dists, n_clusters, sizeof(cluster_dist), compare_cluster_dist);
    }

    // 3. Scan vectors in top nprobe clusters
    knn_entry* top_k = (knn_entry*)enif_alloc(k * sizeof(knn_entry));
    int top_k_len = 0;
    float max_dist = 0.0f;
    int max_idx = 0;

    for (int p = 0; p < actual_nprobe; p++) {
        int c = cluster_dists[p].idx;
        int start = bucket_starts[c];
        int end = bucket_starts[c + 1];

        for (int i = start; i < end; i++) {
            float* vec = vectors + i * DIMS;
            float d2 = dist_sq_avx_8d(query, vec);
            float dist = sqrtf(d2);
            unsigned char label = labels[i];

            insert_top_k(top_k, &top_k_len, &max_idx, &max_dist, dist, label, k);
        }

        // Early termination
        if (top_k_len == k && p + 1 < actual_nprobe) {
            float next_cluster_dist = sqrtf(cluster_dists[p + 1].dist);
            if (next_cluster_dist > max_dist * 1.5) {
                break;
            }
        }
    }

    // 4. Build result list
    ERL_NIF_TERM result = enif_make_list(env, 0);
    for (int i = 0; i < top_k_len; i++) {
        ERL_NIF_TERM dist_term = enif_make_double(env, top_k[i].dist);
        ERL_NIF_TERM label_term = enif_make_uint(env, top_k[i].label);
        ERL_NIF_TERM tuple = enif_make_tuple2(env, dist_term, label_term);
        result = enif_make_list_cell(env, tuple, result);
    }

    enif_free(cluster_dists);
    enif_free(top_k);
    return result;
}

static ERL_NIF_TERM project_svd(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary query_bin, matrix_bin;

    if (!enif_inspect_binary(env, argv[0], &query_bin)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[1], &matrix_bin)) return enif_make_badarg(env);

    float* query = (float*)query_bin.data;
    float* matrix = (float*)matrix_bin.data;

    float result[DIMS] = {0};
    for (int j = 0; j < DIMS; j++) {
        float sum = 0.0f;
        for (int i = 0; i < 14; i++) {
            sum += query[i] * matrix[j * 14 + i];
        }
        result[j] = sum;
    }

    ERL_NIF_TERM bin;
    unsigned char* data = enif_make_new_binary(env, DIMS * 4, &bin);
    memcpy(data, result, DIMS * 4);

    return bin;
}

static ErlNifFunc nif_funcs[] = {
    {"knn_search_ivf", 8, knn_search_ivf},
    {"project_svd", 2, project_svd}
};

ERL_NIF_INIT(Elixir.RinhaFraud.KnnNif, nif_funcs, NULL, NULL, NULL, NULL)
