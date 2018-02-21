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

/* This file contains the implementation of the BLAS-3 function ssymm */

#include "ssymm_common.h"  /* shared between sgemm.cu and fast_sgemm.cu */

typedef void (*pf) (struct cublasSsymmParams parms);

static pf ssymm_hw[16] = {
    ssymm_main_hw_lo_right,
    ssymm_main_hw_lo_left,
    ssymm_main_hw_up_right,
    ssymm_main_hw_up_left,
    fast_ssymm_main_hw_lo_right,
    fast_ssymm_main_hw_lo_left,
    fast_ssymm_main_hw_up_right,
    fast_ssymm_main_hw_up_left,
    ssymm_main_hw_lo_right_fulltile,
    ssymm_main_hw_lo_left_fulltile,
    ssymm_main_hw_up_right_fulltile,
    ssymm_main_hw_up_left_fulltile,
    fast_ssymm_main_hw_lo_right_fulltile,
    fast_ssymm_main_hw_lo_left_fulltile,
    fast_ssymm_main_hw_up_right_fulltile,
    fast_ssymm_main_hw_up_left_fulltile,
};

static pf ssymm_sw[16] = {
    ssymm_main_sw_lo_right,
    ssymm_main_sw_lo_left,
    ssymm_main_sw_up_right,
    ssymm_main_sw_up_left,
    fast_ssymm_main_sw_lo_right,
    fast_ssymm_main_sw_lo_left,
    fast_ssymm_main_sw_up_right,
    fast_ssymm_main_sw_up_left,
    ssymm_main_sw_lo_right_fulltile,
    ssymm_main_sw_lo_left_fulltile,
    ssymm_main_sw_up_right_fulltile,
    ssymm_main_sw_up_left_fulltile,
    fast_ssymm_main_sw_lo_right_fulltile,
    fast_ssymm_main_sw_lo_left_fulltile,
    fast_ssymm_main_sw_up_right_fulltile,
    fast_ssymm_main_sw_up_left_fulltile,
};

/*
 * void 
 * cublasSsymm (char side, char uplo, int m, int n, float alpha, 
 *              const float *A, int lda, const float *B, int ldb, 
 *              float beta, float *C, int ldc);
 *
 * performs one of the matrix-matrix operations
 * 
 *   C = alpha * A * B + beta * C, or 
 *   C = alpha * B * A + beta * C,
 *
 * where alpha and beta are single precision scalars, A is a symmetric matrix
 * consisting of single precision elements and stored in either lower or upper 
 * storage mode, and B and C are m x n matrices consisting of single precision
 * elements.
 *
 * Input
 * -----
 * side   specifies whether the symmetric matrix A appears on the left side 
 *        hand side or right hand side of matrix B, as follows. If side == 'L' 
 *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r', 
 *        then C = alpha * B * A + beta * C.
 * uplo   specifies whether the symmetric matrix A is stored in upper or lower 
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper 
 *        triangular part of the symmetric matrix is to be referenced, and the 
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the 
 *        lower triangular part of the symmetric matrix is to be referenced, 
 *        and the elements of the strictly upper triangular part are to be 
 *        infered from those in the lower triangular part.
 * m      specifies the number of rows of the matrix C, and the number of rows
 *        of matrix B. It also specifies the dimensions of symmetric matrix A 
 *        when side == 'L' or 'l'. m must be at least zero.
 * n      specifies the number of columns of the matrix C, and the number of 
 *        columns of matrix B. It also specifies the dimensions of symmetric 
 *        matrix A when side == 'R' or 'r'. n must be at least zero.
 * alpha  single precision scalar multiplier applied to A * B, or B * A
 * A      single precision array of dimensions (lda, ka), where ka is m when 
 *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the 
 *        leading m x m part of array A must contain the symmetric matrix, 
 *        such that when uplo == 'U' or 'u', the leading m x m part stores the 
 *        upper triangular part of the symmetric matrix, and the strictly lower
 *        triangular part of A is not referenced, and when uplo == 'U' or 'u', 
 *        the leading m x m part stores the lower triangular part of the 
 *        symmetric matrix and the strictly upper triangular part is not 
 *        referenced. If side == 'R' or 'r' the leading n x n part of array A 
 *        must contain the symmetric matrix, such that when uplo == 'U' or 'u',
 *        the leading n x n part stores the upper triangular part of the 
 *        symmetric matrix and the strictly lower triangular part of A is not 
 *        referenced, and when uplo == 'U' or 'u', the leading n x n part 
 *        stores the lower triangular part of the symmetric matrix and the 
 *        strictly upper triangular part is not referenced.
 * lda    leading dimension of A. When side == 'L' or 'l', it must be at least 
 *        max(1, m) and at least max(1, n) otherwise.
 * B      single precision array of dimensions (ldb, n). On entry, the leading
 *        m x n part of the array contains the matrix B.
 * ldb    leading dimension of B. It must be at least max (1, m).
 * beta   single precision scalar multiplier applied to C. If beta is zero, C 
 *        does not have to be a valid input
 * C      single precision array of dimensions (ldc, n)
 * ldc    leading dimension of C. Must be at least max(1, m)
 *
 * Output
 * ------
 * C      updated according to C = alpha * A * B + beta * C, or C = alpha * 
 *        B * A + beta * C
 *
 * Reference: http://www.netlib.org/blas/ssymm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSsymm (char side, char uplo, int m, int n,
                                     float alpha, const float *A, int lda,
                                     const float *B, int ldb, float beta,
                                     float *C, int ldc)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasSsymmParams params;
    cudaError_t cudaStat;
    int nrowa, upper, lside;
    int info = 0;
    int usePureHwStepper;
    int useFastImul;
    int fullTilesOnly;
    int funcIdx;
    dim3 ctaDimsHw (((n+TILE_DIM-1)/TILE_DIM), ((m+TILE_DIM-1)/TILE_DIM));
    dim3 ctaDimsSw (CUBLAS_SSYMM_GRIDW, CUBLAS_SSYMM_GRIDH);

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }

    lside = toupper (side) == 'L';
    upper = toupper (uplo) == 'U';
    nrowa = lside ? m : n;

    /* Test the input parameters. */

    info = 0;
    if ((!lside) && (!(toupper (side) == 'R'))) {
        info = 1;
    }
    else if ((!upper) && (!(toupper (uplo) == 'L'))) {
        info = 2;
    }
    else if (m < 0) {
        info = 3;
    }
    else if (n < 0) {
        info = 4;
    }
    else if (lda < imax (1, nrowa)) {
        info = 7;
    }
    else if (ldb < imax (1, m)) {
        info = 9;
    }
    else if (ldc < imax (1, m)) {
        info = 12;
    }
    if (info) {
        cublasXerbla ("SSYMM ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((m == 0) || (n == 0) || ((alpha == 0.0f) && (beta == 1.0f))) {
        return;
    }

    memset (&params, 0, sizeof(params));
    params.lside = lside;
    params.upper = upper;
    params.k = nrowa;
    params.m = m;
    params.n = n;
    params.alpha = alpha;
    params.A = lside ? A : B;
    params.lda = lside ? lda : ldb;
    params.B = lside ? B : A;
    params.ldb = lside ? ldb : lda;
    params.beta = beta;
    params.C = C;
    params.ldc =ldc;

    /* choose HW-only stepping if dimensions of result matrix do not exceed the
     * maximum CTA grid dimensions.
     */
    usePureHwStepper = ((params.m < (CUBLAS_CTA_MAX_DIM * TILE_DIM)) &&
                        (params.n < (CUBLAS_CTA_MAX_DIM * TILE_DIM)));

    /* we can eliminate checking for endcases if we know all tiles are fully
     * populated. Important benchmark case!
     */
    fullTilesOnly = (((params.m % TILE_DIM) == 0) &&
                     ((params.n % TILE_DIM) == 0) &&
                     ((params.k % TILE_DIM) == 0));

    /* choose version using 24-bit multiplies if all dimensions are less than
     * 2001, so we can guarantee that no multiplication result exceeds (2000 *
     * 2000 * 4) < 2^24.
     */
    useFastImul = ((params.lda <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                   (params.ldb <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                   (params.ldc <= CUBLAS_FASTIMUL_F_MAX_DIM) &&
                   (params.m <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                   (params.n <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                   (params.k <= CUBLAS_FASTIMUL_F_MAX_DIM));

    funcIdx = ((fullTilesOnly << 3) | (useFastImul << 2) | 
               (params.upper << 1) | params.lside);

    cudaStat = cudaGetLastError(); /* clear error status */
    if (usePureHwStepper) {
        ssymm_hw[funcIdx]<<<ctaDimsHw,CUBLAS_SSYMM_THREAD_COUNT>>>(params);
    } else {
        ssymm_sw[funcIdx]<<<ctaDimsSw,CUBLAS_SSYMM_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */
    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

__global__ void ssymm_main_hw_lo_right (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             0
#define LSIDE             0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void ssymm_main_hw_lo_left (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             0
#define LSIDE             1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void ssymm_main_hw_up_right (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             1
#define LSIDE             0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void ssymm_main_hw_up_left (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             1
#define LSIDE             1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void ssymm_main_sw_lo_right (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             0
#define LSIDE             0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void ssymm_main_sw_lo_left (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             0
#define LSIDE             1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void ssymm_main_sw_up_right (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             1
#define LSIDE             0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void ssymm_main_sw_up_left (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             1
#define LSIDE             1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void ssymm_main_hw_lo_right_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             0
#define LSIDE             0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void ssymm_main_hw_lo_left_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             0
#define LSIDE             1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void ssymm_main_hw_up_right_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             1
#define LSIDE             0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void ssymm_main_hw_up_left_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             1
#define LSIDE             1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void ssymm_main_sw_lo_right_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             0
#define LSIDE             0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void ssymm_main_sw_lo_left_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             0
#define LSIDE             1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void ssymm_main_sw_up_right_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             1
#define LSIDE             0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void ssymm_main_sw_up_left_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             1
#define LSIDE             1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}
