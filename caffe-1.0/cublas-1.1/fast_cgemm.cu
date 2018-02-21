/*
 * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  
 *
 * This software and the information contained herein is being provided 
 * under the terms and conditions of a Source Code License Agreement.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* This file contains the implementation of the BLAS-3 function cgemm */

/* Use square 16x16 tiles to access and cache portions of source matrices A,B 
 * and result matrix C
 */
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "cublasP.h"
#define TILE_DIM_LOG           (4)
#define TILE_DIM               (1 << TILE_DIM_LOG)
#define TILE_SIZE              (TILE_DIM*TILE_DIM)
#define SUP_TILE_DIM           (TILE_DIM*CUBLAS_CGEMM_GRIDW)
#define COL_INCR               (CUBLAS_CGEMM_THREAD_COUNT/TILE_DIM)
#define C_ELEMS_PER_THREAD     (TILE_SIZE/CUBLAS_CGEMM_THREAD_COUNT)
#define A_ELEMS_PER_THREAD     (TILE_SIZE/CUBLAS_CGEMM_THREAD_COUNT)
#define B_ELEMS_PER_THREAD     (TILE_SIZE/CUBLAS_CGEMM_THREAD_COUNT)

__global__ void fast_cgemm_1_main_hw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_2_main_hw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_3_main_hw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_4_main_hw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_5_main_hw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_6_main_hw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_7_main_hw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_8_main_hw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_9_main_hw_gld (struct cublasCgemmParams parms);

__global__ void fast_cgemm_1_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_2_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_3_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_4_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_5_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_6_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_7_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_8_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_9_main_hw_gld_fulltile (struct cublasCgemmParams parms);

__global__ void fast_cgemm_1_main_hw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_2_main_hw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_3_main_hw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_4_main_hw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_5_main_hw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_6_main_hw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_7_main_hw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_8_main_hw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_9_main_hw_tex (struct cublasCgemmParams parms);

__global__ void fast_cgemm_1_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_2_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_3_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_4_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_5_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_6_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_7_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_8_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_9_main_hw_tex_fulltile (struct cublasCgemmParams parms);

__global__ void fast_cgemm_1_main_sw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_2_main_sw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_3_main_sw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_4_main_sw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_5_main_sw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_6_main_sw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_7_main_sw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_8_main_sw_gld (struct cublasCgemmParams parms);
__global__ void fast_cgemm_9_main_sw_gld (struct cublasCgemmParams parms);

__global__ void fast_cgemm_1_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_2_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_3_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_4_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_5_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_6_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_7_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_8_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_9_main_sw_gld_fulltile (struct cublasCgemmParams parms);

__global__ void fast_cgemm_1_main_sw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_2_main_sw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_3_main_sw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_4_main_sw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_5_main_sw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_6_main_sw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_7_main_sw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_8_main_sw_tex (struct cublasCgemmParams parms);
__global__ void fast_cgemm_9_main_sw_tex (struct cublasCgemmParams parms);

__global__ void fast_cgemm_1_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_2_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_3_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_4_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_5_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_6_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_7_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_8_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void fast_cgemm_9_main_sw_tex_fulltile (struct cublasCgemmParams parms);

texture<float2> texA;
texture<float2> texB;

__shared__ float AA_r[(TILE_DIM+1)*TILE_DIM];/*pad to elim GRF bank conflicts*/
__shared__ float BB_r[(TILE_DIM+1)*TILE_DIM];/*pad to elim GRF bank conflicts*/
__shared__ float AA_i[(TILE_DIM+1)*TILE_DIM];/*pad to elim GRF bank conflicts*/
__shared__ float BB_i[(TILE_DIM+1)*TILE_DIM];/*pad to elim GRF bank conflicts*/

typedef void (*pf) (struct cublasCgemmParams parms);

static pf cgemm_hw[64] = {
    fast_cgemm_9_main_hw_gld, /* C = alpha*transpose(A)*transpose(B) + beta*C */
    fast_cgemm_8_main_hw_gld, /* C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
    fast_cgemm_7_main_hw_gld, /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
    fast_cgemm_6_main_hw_gld, /* C = alpha*conjg(transpose(A))*conjg(transpose(B))+beta*C */
    fast_cgemm_3_main_hw_gld, /* C = alpha*transpose(A)*B + beta*C */
    fast_cgemm_3_main_hw_gld, /* C = alpha*transpose(A)*B + beta*C */
    fast_cgemm_2_main_hw_gld, /* C = alpha*conj(transpose(A))*B + beta*C */
    fast_cgemm_2_main_hw_gld, /* C = alpha*conj(transpose(A))*B + beta*C */
    fast_cgemm_5_main_hw_gld, /* C = alpha*A*transpose(B) + beta*C */
    fast_cgemm_4_main_hw_gld, /* C = alpha*A*conj(transpose(B)) + beta*C */
    fast_cgemm_5_main_hw_gld, /* C = alpha*A*transpose(B) + beta*C */
    fast_cgemm_4_main_hw_gld, /* C = alpha*A*conj(transpose(B)) + beta*C */
    fast_cgemm_1_main_hw_gld, /* C = alpha*A*B + beta*C */
    fast_cgemm_1_main_hw_gld, /* C = alpha*A*B + beta*C */
    fast_cgemm_1_main_hw_gld, /* C = alpha*A*B + beta*C */
    fast_cgemm_1_main_hw_gld, /* C = alpha*A*B + beta*C */
    fast_cgemm_9_main_hw_gld_fulltile,
    fast_cgemm_8_main_hw_gld_fulltile,
    fast_cgemm_7_main_hw_gld_fulltile,
    fast_cgemm_6_main_hw_gld_fulltile,
    fast_cgemm_3_main_hw_gld_fulltile,
    fast_cgemm_3_main_hw_gld_fulltile,
    fast_cgemm_2_main_hw_gld_fulltile,
    fast_cgemm_2_main_hw_gld_fulltile,
    fast_cgemm_5_main_hw_gld_fulltile,
    fast_cgemm_4_main_hw_gld_fulltile,
    fast_cgemm_5_main_hw_gld_fulltile,
    fast_cgemm_4_main_hw_gld_fulltile,
    fast_cgemm_1_main_hw_gld_fulltile,
    fast_cgemm_1_main_hw_gld_fulltile,
    fast_cgemm_1_main_hw_gld_fulltile,
    fast_cgemm_1_main_hw_gld_fulltile,
    fast_cgemm_9_main_hw_tex,
    fast_cgemm_8_main_hw_tex,
    fast_cgemm_7_main_hw_tex,
    fast_cgemm_6_main_hw_tex,
    fast_cgemm_3_main_hw_tex,
    fast_cgemm_3_main_hw_tex,
    fast_cgemm_2_main_hw_tex,
    fast_cgemm_2_main_hw_tex,
    fast_cgemm_5_main_hw_tex,
    fast_cgemm_4_main_hw_tex,
    fast_cgemm_5_main_hw_tex,
    fast_cgemm_4_main_hw_tex,
    fast_cgemm_1_main_hw_tex,
    fast_cgemm_1_main_hw_tex,
    fast_cgemm_1_main_hw_tex,
    fast_cgemm_1_main_hw_tex,
    fast_cgemm_9_main_hw_tex_fulltile,
    fast_cgemm_8_main_hw_tex_fulltile,
    fast_cgemm_7_main_hw_tex_fulltile,
    fast_cgemm_6_main_hw_tex_fulltile,
    fast_cgemm_3_main_hw_tex_fulltile,
    fast_cgemm_3_main_hw_tex_fulltile,
    fast_cgemm_2_main_hw_tex_fulltile,
    fast_cgemm_2_main_hw_tex_fulltile,
    fast_cgemm_5_main_hw_tex_fulltile,
    fast_cgemm_4_main_hw_tex_fulltile,
    fast_cgemm_5_main_hw_tex_fulltile,
    fast_cgemm_4_main_hw_tex_fulltile,
    fast_cgemm_1_main_hw_tex_fulltile,
    fast_cgemm_1_main_hw_tex_fulltile,
    fast_cgemm_1_main_hw_tex_fulltile,
    fast_cgemm_1_main_hw_tex_fulltile
};

static pf cgemm_sw[64] = {
    fast_cgemm_9_main_sw_gld, /* C = alpha*transpose(A)*transpose(B) + beta*C */
    fast_cgemm_8_main_hw_gld, /* C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
    fast_cgemm_7_main_sw_gld, /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
    fast_cgemm_6_main_sw_gld, /* C = alpha*conjg(transpose(A))*conjg(transpose(B))+beta*C */
    fast_cgemm_3_main_sw_gld, /* C = alpha*transpose(A)*B + beta*C */
    fast_cgemm_3_main_sw_gld, /* C = alpha*transpose(A)*B + beta*C */
    fast_cgemm_2_main_sw_gld, /* C = alpha*conj(transpose(A))*B + beta*C */
    fast_cgemm_2_main_sw_gld, /* C = alpha*conj(transpose(A))*B + beta*C */
    fast_cgemm_5_main_sw_gld, /* C = alpha*A*transpose(B) + beta*C */
    fast_cgemm_4_main_sw_gld, /* C = alpha*A*conj(transpose(B)) + beta*C */
    fast_cgemm_5_main_sw_gld, /* C = alpha*A*transpose(B) + beta*C */
    fast_cgemm_4_main_sw_gld, /* C = alpha*A*conj(transpose(B)) + beta*C */
    fast_cgemm_1_main_sw_gld, /* C = alpha*A*B + beta*C */
    fast_cgemm_1_main_sw_gld, /* C = alpha*A*B + beta*C */
    fast_cgemm_1_main_sw_gld, /* C = alpha*A*B + beta*C */
    fast_cgemm_1_main_sw_gld, /* C = alpha*A*B + beta*C */
    fast_cgemm_9_main_sw_gld_fulltile,
    fast_cgemm_8_main_sw_gld_fulltile,
    fast_cgemm_7_main_sw_gld_fulltile,
    fast_cgemm_6_main_sw_gld_fulltile,
    fast_cgemm_3_main_sw_gld_fulltile,
    fast_cgemm_3_main_sw_gld_fulltile,
    fast_cgemm_2_main_sw_gld_fulltile,
    fast_cgemm_2_main_sw_gld_fulltile,
    fast_cgemm_5_main_sw_gld_fulltile,
    fast_cgemm_4_main_sw_gld_fulltile,
    fast_cgemm_5_main_sw_gld_fulltile,
    fast_cgemm_4_main_sw_gld_fulltile,
    fast_cgemm_1_main_sw_gld_fulltile,
    fast_cgemm_1_main_sw_gld_fulltile,
    fast_cgemm_1_main_sw_gld_fulltile,
    fast_cgemm_1_main_sw_gld_fulltile,
    fast_cgemm_9_main_sw_tex,
    fast_cgemm_8_main_sw_tex,
    fast_cgemm_7_main_sw_tex,
    fast_cgemm_6_main_sw_tex,
    fast_cgemm_3_main_sw_tex,
    fast_cgemm_3_main_sw_tex,
    fast_cgemm_2_main_sw_tex,
    fast_cgemm_2_main_sw_tex,
    fast_cgemm_5_main_sw_tex,
    fast_cgemm_4_main_sw_tex,
    fast_cgemm_5_main_sw_tex,
    fast_cgemm_4_main_sw_tex,
    fast_cgemm_1_main_sw_tex,
    fast_cgemm_1_main_sw_tex,
    fast_cgemm_1_main_sw_tex,
    fast_cgemm_1_main_sw_tex,
    fast_cgemm_9_main_sw_tex_fulltile,
    fast_cgemm_8_main_sw_tex_fulltile,
    fast_cgemm_7_main_sw_tex_fulltile,
    fast_cgemm_6_main_sw_tex_fulltile,
    fast_cgemm_3_main_sw_tex_fulltile,
    fast_cgemm_3_main_sw_tex_fulltile,
    fast_cgemm_2_main_sw_tex_fulltile,
    fast_cgemm_2_main_sw_tex_fulltile,
    fast_cgemm_5_main_sw_tex_fulltile,
    fast_cgemm_4_main_sw_tex_fulltile,
    fast_cgemm_5_main_sw_tex_fulltile,
    fast_cgemm_4_main_sw_tex_fulltile,
    fast_cgemm_1_main_sw_tex_fulltile,
    fast_cgemm_1_main_sw_tex_fulltile,
    fast_cgemm_1_main_sw_tex_fulltile,
    fast_cgemm_1_main_sw_tex_fulltile
};

void cublasFastCgemm (struct cublasContext *ctx, char transa, char transb, 
                      int m, int n, int k, cuComplex alpha, const cuComplex *A,
                      int lda, const cuComplex *B, int ldb, cuComplex beta, 
                      cuComplex *C, int ldc)
{
    struct cublasCgemmParams params;
    cudaError_t cudaStat;
    int fullTilesOnly;
    size_t texAOfs = 0;
    size_t texBOfs = 0;
    int notransa, notransb, conja, conjb;
    int useTexture;
    int usePureHwStepper;
    int funcIdx;
    int sizeA = lda * ((toupper(transa) == 'N') ? k : m);
    int sizeB = ldb * ((toupper(transb) == 'N') ? n : k);   
    dim3 ctaDimsHw (((n+TILE_DIM-1)/TILE_DIM), ((m+TILE_DIM-1)/TILE_DIM));
    dim3 ctaDimsSw (CUBLAS_CGEMM_GRIDW, CUBLAS_CGEMM_GRIDH);

    notransa = toupper(transa) == 'N';
    notransb = toupper(transb) == 'N';
    conja  = toupper(transa) == 'C';
    conjb  = toupper(transb) == 'C';

    /* We can only use texture if the matrices fit into the largest matrix 
     * size supported.
     */
    useTexture = ((sizeA < CUBLAS_MAX_1DBUF_SIZE) &&
                  (sizeB < CUBLAS_MAX_1DBUF_SIZE));

    /* choose HW-only stepping if dimensions of result matrix do not exceed the
     * maximum CTA grid dimensions.
     */
    usePureHwStepper = ((m < (CUBLAS_CTA_MAX_DIM * TILE_DIM)) &&
                        (n < (CUBLAS_CTA_MAX_DIM * TILE_DIM)));

    /* we can eliminate checking for endcases if we know all tiles are fully
     * populated. Important benchmark case!
     */
    fullTilesOnly = (((m % TILE_DIM) == 0) &&
                     ((n % TILE_DIM) == 0) &&
                     ((k % TILE_DIM) == 0));

    /* currently, texture binding is expensive, so using texture fetches
     * is a net negative for small cases. For matrices where each row is
     * aligned, GLD coalesces nicely and is faster, so don't use texture.
     */
    if (((m * k + k * n) < 32768) ||  /* empirically determined cutoff */
        (!(((ptrdiff_t) A) % CUBLAS_WORD_ALIGN) && 
         !(((ptrdiff_t) B) % CUBLAS_WORD_ALIGN) &&
         !(lda % (CUBLAS_WORD_ALIGN / sizeof(A[0]))) &&
         !(ldb % (CUBLAS_WORD_ALIGN / sizeof(B[0]))))) {
        useTexture = 0;
    }
    
    if (useTexture){
        if ((cudaStat=cudaBindTexture (&texAOfs,texA,A,sizeA*sizeof(A[0]))) != cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_MAPPING_ERROR);
            return;
        }
        if ((cudaStat=cudaBindTexture (&texBOfs,texB,B,sizeB*sizeof(B[0]))) != cudaSuccess) {
            cudaUnbindTexture (texA);
            cublasSetError (ctx, CUBLAS_STATUS_MAPPING_ERROR);
            return;
        }
        texAOfs /= sizeof(A[0]);
        texBOfs /= sizeof(B[0]);
    }

    memset (&params, 0, sizeof(params));
    params.m = m;
    params.n = n;
    params.k = k;
    params.alpha = alpha;
    params.A = A;
    params.lda = lda;
    params.B = B;
    params.ldb = ldb;
    params.beta = beta;
    params.C = C;
    params.ldc =ldc;
    params.texAOfs = (int)texAOfs;
    params.texBOfs = (int)texBOfs;
    
    funcIdx = ((useTexture << 5) | (fullTilesOnly << 4) | (notransa << 3) | 
               (notransb << 2) | (conja << 1) | (conjb << 0));

    cudaStat = cudaGetLastError(); /* clear error status */
    if (usePureHwStepper) {
        cgemm_hw[funcIdx]<<<ctaDimsHw,CUBLAS_CGEMM_THREAD_COUNT>>>(params);
    } else {
        cgemm_sw[funcIdx]<<<ctaDimsSw,CUBLAS_CGEMM_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */
    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
    if (useTexture) {
        if ((cudaStat = cudaUnbindTexture (texA)) != cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if ((cudaStat = cudaUnbindTexture (texB)) != cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR);
        }
    }
}

__global__ void fast_cgemm_1_main_hw_gld (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_2_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_3_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_4_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_5_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_6_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_7_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_8_main_hw_gld (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_9_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_1_main_hw_gld_fulltile (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_2_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_3_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_4_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_5_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_6_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_7_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_8_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_9_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

/* --------------------------------------------------------------- */

__global__ void fast_cgemm_1_main_hw_tex (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_2_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_3_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_4_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_5_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_6_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_7_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_8_main_hw_tex (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_9_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_1_main_hw_tex_fulltile (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_2_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_3_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_4_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_5_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_6_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_7_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_8_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_9_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_1_main_sw_gld (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_2_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_3_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_4_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_5_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_6_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_7_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_8_main_sw_gld (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_9_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_1_main_sw_gld_fulltile (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_2_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_3_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_4_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_5_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_6_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_7_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_8_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_9_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_1_main_sw_tex (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_2_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_3_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_4_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_5_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_6_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_7_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_8_main_sw_tex (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_9_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_1_main_sw_tex_fulltile (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_2_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_3_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_4_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_5_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_6_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_7_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void fast_cgemm_8_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void fast_cgemm_9_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}
