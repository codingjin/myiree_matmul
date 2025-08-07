// Matrix Multiplication Implementation for x86_64
// Compile with: clang -march=native -mavx512f avx512_16.c -o avx512_16

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 1

#define WARMUP 10
#define RUN 10
#define MAXTHREADNUM 32

#define UK_PREFETCH_RO(ptr, hint) __builtin_prefetch((ptr), /*rw=*/0, hint)
#define UK_PREFETCH_RW(ptr, hint) __builtin_prefetch((ptr), /*rw=*/1, hint)
// This is how the 0--3 locality hint is interpreted by both Clang and GCC on
// both arm64 and x86-64.
#define UK_PREFETCH_LOCALITY_NONE 0  // No locality. Data accessed once.
#define UK_PREFETCH_LOCALITY_L3 1    // Some locality. Try to keep in L3.
#define UK_PREFETCH_LOCALITY_L2 2    // More locality. Try to keep in L2.
#define UK_PREFETCH_LOCALITY_L1 3    // Most locality. Try to keep in L1.

// Structure to hold tiled matrix data
typedef struct {
    float* data;
    int rows;
    int cols;
    int tile_rows;
    int tile_cols;
} TiledMatrix;
TiledMatrix* pack_matrix(const float* src, int rows, int cols, int tile_size_m, int tile_size_n);
void unpack_matrix(float* dst, const TiledMatrix* tiled, int tile_size_m, int tile_size_n);
void matmul_reference(float* C, const float* A, const float* B, int M, int N, int K);
void init_matrix(float* mat, int size);
int check_result(const float* mat1, const float* mat2, int size, float tolerance);
void iree_mmt4d(TiledMatrix* acc, const TiledMatrix* lhs, const TiledMatrix* rhs);


int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: ./avx512_16_p M N K\n");
        exit(1);
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    printf("\n\nMatrix multiplication: %dx%d * %dx%d = %dx%d\n", M, K, K, N, M, N);
    
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C_ref = (float*)malloc(M * N * sizeof(float));
    float* C_iree = (float*)malloc(M * N * sizeof(float));

    double flops = 2.0 * M * N * K;

    // Initialize matrices
    srand(137);
    init_matrix(A, M * K);
    init_matrix(B, K * N);
    
    // Reference implementation
    matmul_reference(C_ref, A, B, M, N, K);

    for (int tn = 2; tn <= MAXTHREADNUM; ++tn) {
        omp_set_num_threads(tn);
        // OpenMP mode
        for (int i = 0; i < WARMUP; ++i) {
            memset(C_iree, 0, M * N * sizeof(float));
            // Pack matrices into tiled format
            TiledMatrix* A_tiled = pack_matrix(A, M, K, TILE_M, TILE_K);
            TiledMatrix* B_tiled = pack_matrix(B, K, N, TILE_K, TILE_N);
            TiledMatrix* C_tiled = pack_matrix(C_iree, M, N, TILE_M, TILE_N);
            iree_mmt4d(C_tiled, A_tiled, B_tiled);
            unpack_matrix(C_iree, C_tiled, TILE_M, TILE_N);
            free(A_tiled->data);
            free(A_tiled);
            free(B_tiled->data);
            free(B_tiled);
            free(C_tiled->data);
            free(C_tiled);
        }
        
        clock_t t0, t1;
        double time[RUN];
        double avg_time = 0.0;
        for (int i = 0; i < RUN; ++i) {
            memset(C_iree, 0, M * N * sizeof(float));
            TiledMatrix* A_tiled = pack_matrix(A, M, K, TILE_M, TILE_K);
            TiledMatrix* B_tiled = pack_matrix(B, K, N, TILE_K, TILE_N);
            TiledMatrix* C_tiled = pack_matrix(C_iree, M, N, TILE_M, TILE_N);

            t0 = clock();
            iree_mmt4d(C_tiled, A_tiled, B_tiled);
            t1 = clock();
            unpack_matrix(C_iree, C_tiled, TILE_M, TILE_N);
            time[i] = (double)(t1 - t0) / CLOCKS_PER_SEC;
            free(A_tiled->data);
            free(A_tiled);
            free(B_tiled->data);
            free(B_tiled);
            free(C_tiled->data);
            free(C_tiled);
        }
        for (int i = 0; i < RUN; ++i) {
            avg_time += time[i];
        }
        avg_time /= RUN;

        // Verify results
        if (!check_result(C_ref, C_iree, M * N, 1e-5)) {
            printf("OpenMp mode (ThreadNum=%d) Results do not match!\n", tn);
            exit(1);
        }
        printf("OpenMP ThreadNum=%d : %.0f GFLOPS\n", tn, flops / avg_time / 1e9);
    }
    // Cleanup
    free(A);
    free(B);
    free(C_ref);
    free(C_iree);
    
    return 0;
}


void iree_mmt4d(TiledMatrix* acc, const TiledMatrix* lhs, const TiledMatrix* rhs) {
    // micro-kernel: 16x16x1
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < lhs->tile_rows; m++) {
        for (int n = 0; n < rhs->tile_cols; n++) {
            const float* lhs_tile = &lhs->data[(m * lhs->tile_cols) * TILE_M];
            const float* rhs_tile = &rhs->data[n * TILE_N];
            float* acc_tile = &acc->data[(m * acc->tile_cols + n) * TILE_M * TILE_N];
            UK_PREFETCH_RW(acc_tile, UK_PREFETCH_LOCALITY_L3);
            UK_PREFETCH_RO(lhs_tile, UK_PREFETCH_LOCALITY_L1);
            UK_PREFETCH_RO(rhs_tile, UK_PREFETCH_LOCALITY_L1);

            __m512 acc[16];
            #pragma clang loop unroll(full)
            for (int i = 0; i < 16; i++) {
                acc[i] = _mm512_setzero_ps();
            }

            for (int k = 0; k < lhs->tile_cols; ++k) {
                __m512 rhs_loaded = _mm512_loadu_ps(rhs_tile);
                _mm_prefetch((const char*)(rhs_tile + 128 * rhs->tile_cols), _MM_HINT_T0);
                rhs_tile += rhs->tile_cols << 4;

                #pragma clang loop unroll(full)
                for (int i = 0; i < 16; ++i) {
                    acc[i] = _mm512_fmadd_ps(rhs_loaded, _mm512_set1_ps(lhs_tile[i]), acc[i]);
                }
                _mm_prefetch((const char*)(lhs_tile + 128), _MM_HINT_T0);
                lhs_tile += 16;
            }

            // Store results back to accumulator
            #pragma clang loop unroll(full)
            for (int i = 0; i < 16; i++) {
                _mm512_storeu_ps(acc_tile + (i << 4), acc[i]);
            }
        }
    }
}


// Pack matrix into tiled layout
TiledMatrix* pack_matrix(const float* src, int rows, int cols, int tile_size_m, int tile_size_n) {
    TiledMatrix* tiled = (TiledMatrix*)malloc(sizeof(TiledMatrix));
    
    // Calculate number of tiles (with padding)
    tiled->tile_rows = (rows + tile_size_m - 1) / tile_size_m;
    tiled->tile_cols = (cols + tile_size_n - 1) / tile_size_n;
    tiled->rows = rows;
    tiled->cols = cols;
    
    // Allocate tiled data
    int total_elements = tiled->tile_rows * tiled->tile_cols * tile_size_m * tile_size_n;
    tiled->data = (float*)aligned_alloc(64, total_elements * sizeof(float));
    memset(tiled->data, 0, total_elements * sizeof(float));
    
    // Pack data into tiles
    for (int tile_i = 0; tile_i < tiled->tile_rows; tile_i++) {
        for (int tile_j = 0; tile_j < tiled->tile_cols; tile_j++) {
            float* tile_ptr = &tiled->data[(tile_i * tiled->tile_cols + tile_j) * tile_size_m * tile_size_n];
            
            for (int i = 0; i < tile_size_m && tile_i * tile_size_m + i < rows; i++) {
                for (int j = 0; j < tile_size_n && tile_j * tile_size_n + j < cols; j++) {
                    int src_idx = (tile_i * tile_size_m + i) * cols + (tile_j * tile_size_n + j);
                    int tile_idx = i * tile_size_n + j;
                    tile_ptr[tile_idx] = src[src_idx];
                }
            }
        }
    }
    
    return tiled;
}

// Unpack tiled matrix back to regular layout
void unpack_matrix(float* dst, const TiledMatrix* tiled, int tile_size_m, int tile_size_n) {
    for (int tile_i = 0; tile_i < tiled->tile_rows; tile_i++) {
        for (int tile_j = 0; tile_j < tiled->tile_cols; tile_j++) {
            const float* tile_ptr = &tiled->data[(tile_i * tiled->tile_cols + tile_j) * tile_size_m * tile_size_n];
            
            for (int i = 0; i < tile_size_m && tile_i * tile_size_m + i < tiled->rows; i++) {
                for (int j = 0; j < tile_size_n && tile_j * tile_size_n + j < tiled->cols; j++) {
                    int dst_idx = (tile_i * tile_size_m + i) * tiled->cols + (tile_j * tile_size_n + j);
                    int tile_idx = i * tile_size_n + j;
                    dst[dst_idx] = tile_ptr[tile_idx];
                }
            }
        }
    }
}


// Standard matrix multiplication for verification
void matmul_reference(float* C, const float* A, const float* B, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// Initialize matrix with random values
void init_matrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (float)rand() / (float)RAND_MAX;
    }
}

// Check if two matrices are close enough
int check_result(const float* mat1, const float* mat2, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        float diff = fabs(mat1[i] - mat2[i]);
        //printf("diff at %d\t : %f\n", i, diff);
        if (diff > tolerance) {
            printf("Mismatch at index %d: %f vs %f (diff: %f)\n", i, mat1[i], mat2[i], diff);
            return 0;
        }
    }
    return 1;
}


