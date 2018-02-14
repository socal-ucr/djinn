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

/* This file contains the implementation of the BLAS-3 function sgemm */

/* Use square 16x16 tiles to access and cache portions of source matrices A,B 
 * and result matrix C
 */
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "cublasP.h"
#define TILE_DIM_LOG    (4)
#define THREAD_COUNT    (CUBLAS_SGEMM_SMALL_THREAD_COUNT)
#include "sgemm_sizing.h"
#include "sgemm_common.h"

texture<float> texA;
texture<float> texB;

__shared__ float AA[(TILE_DIM+1)*TILE_DIM]; /*pad to elim. GRF bank conflicts*/
__shared__ float BB[(TILE_DIM+1)*TILE_DIM]; /*pad to elim. GRF bank conflicts*/

typedef void (*pf) (struct cublasSgemmParams parms);

static pf sgemm[16] = {
    fast_small_sgemm_gld_main_hw_na_nb,
    fast_small_sgemm_gld_main_hw_na_tb,
    fast_small_sgemm_gld_main_hw_ta_nb,
    fast_small_sgemm_gld_main_hw_ta_tb,
    fast_small_sgemm_gld_main_hw_na_nb_fulltile,
    fast_small_sgemm_gld_main_hw_na_tb_fulltile,
    fast_small_sgemm_gld_main_hw_ta_nb_fulltile,
    fast_small_sgemm_gld_main_hw_ta_tb_fulltile,
    fast_small_sgemm_tex_main_hw_na_nb,
    fast_small_sgemm_tex_main_hw_na_tb,
    fast_small_sgemm_tex_main_hw_ta_nb,
    fast_small_sgemm_tex_main_hw_ta_tb,
    fast_small_sgemm_tex_main_hw_na_nb_fulltile,
    fast_small_sgemm_tex_main_hw_na_tb_fulltile,
    fast_small_sgemm_tex_main_hw_ta_nb_fulltile,
    fast_small_sgemm_tex_main_hw_ta_tb_fulltile,
};             

__host__ void cublasSmallSgemm (struct cublasContext *ctx, char transa, 
                                char transb, int m, int n, int k, 
                                float alpha, const float *A, int lda, 
                                const float *B, int ldb, float beta, float *C, 
                                int ldc)
{
    struct cublasSgemmParams params;
    cudaError_t cudaStat;
    int fullTilesOnly;
    size_t texAOfs = 0;
    size_t texBOfs = 0;
    int sizeA = lda * ((toupper(transa) == 'N') ? k : m);
    int sizeB = ldb * ((toupper(transb) == 'N') ? n : k);
    int useTexture;
    int funcIdx;
    dim3 ctaDimsHw (((n+TILE_DIM-1)/TILE_DIM), ((m+TILE_DIM-1)/TILE_DIM));

    useTexture = ((sizeA < CUBLAS_MAX_1DBUF_SIZE) &&
                  (sizeB < CUBLAS_MAX_1DBUF_SIZE));

    /* currently, texture binding is too expensive, so using texture fetches
     * is a net negative for the small cases handled here.
     */
    useTexture = 0;

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
    params.transa = ((toupper(transa) == 'T') || (toupper(transa) == 'C'));
    params.transb = ((toupper(transb) == 'T') || (toupper(transb) == 'C'));
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

    /* we can eliminate checking for endcases if we know all tiles are fully
     * populated. Important benchmark case!
     */
    fullTilesOnly = (((params.m % TILE_DIM) == 0) &&
                     ((params.n % TILE_DIM) == 0) &&
                     ((params.k % TILE_DIM) == 0));

    funcIdx = ((useTexture << 3) | (fullTilesOnly << 2) | 
               (params.transa << 1) | params.transb);

    cudaStat = cudaGetLastError(); /* clear error status */
    sgemm[funcIdx]<<<ctaDimsHw,THREAD_COUNT>>>(params);
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
    
__global__ void fast_small_sgemm_tex_main_hw_na_nb (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            0
#define FAST_IMUL         1
#define USE_TEX           1
#include "sgemm.h"
}

__global__ void fast_small_sgemm_tex_main_hw_na_tb (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define FAST_IMUL         1
#define USE_TEX           1
#include "sgemm.h"
}

__global__ void fast_small_sgemm_tex_main_hw_ta_nb (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define FAST_IMUL         1
#define USE_TEX           1
#include "sgemm.h"
}

__global__ void fast_small_sgemm_tex_main_hw_ta_tb (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define FAST_IMUL         1
#define USE_TEX           1
#include "sgemm.h"
}

__global__ void fast_small_sgemm_tex_main_hw_na_nb_fulltile (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            0
#define FAST_IMUL         1
#define USE_TEX           1
#include "sgemm.h"
}

__global__ void fast_small_sgemm_tex_main_hw_na_tb_fulltile (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define FAST_IMUL         1
#define USE_TEX           1
#include "sgemm.h"
}

__global__ void fast_small_sgemm_tex_main_hw_ta_nb_fulltile (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define FAST_IMUL         1
#define USE_TEX           1
#include "sgemm.h"
}

__global__ void fast_small_sgemm_tex_main_hw_ta_tb_fulltile (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define FAST_IMUL         1
#define USE_TEX           1
#include "sgemm.h"
}
    
__global__ void fast_small_sgemm_gld_main_hw_na_nb (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            0
#define FAST_IMUL         1
#define USE_TEX           0
#include "sgemm.h"
}

__global__ void fast_small_sgemm_gld_main_hw_na_tb (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define FAST_IMUL         1
#define USE_TEX           0
#include "sgemm.h"
}

__global__ void fast_small_sgemm_gld_main_hw_ta_nb (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define FAST_IMUL         1
#define USE_TEX           0
#include "sgemm.h"
}

__global__ void fast_small_sgemm_gld_main_hw_ta_tb (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define FAST_IMUL         1
#define USE_TEX           0
#include "sgemm.h"
}

__global__ void fast_small_sgemm_gld_main_hw_na_nb_fulltile (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            0
#define FAST_IMUL         1
#define USE_TEX           0
#include "sgemm.h"
}

__global__ void fast_small_sgemm_gld_main_hw_na_tb_fulltile (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define FAST_IMUL         1
#define USE_TEX           0
#include "sgemm.h"
}

__global__ void fast_small_sgemm_gld_main_hw_ta_nb_fulltile (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define FAST_IMUL         1
#define USE_TEX           0
#include "sgemm.h"
}

__global__ void fast_small_sgemm_gld_main_hw_ta_tb_fulltile (struct cublasSgemmParams parms)
{
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  FAST_IMUL
#undef  USE_TEX
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define FAST_IMUL         1
#define USE_TEX           0
#include "sgemm.h"
}
