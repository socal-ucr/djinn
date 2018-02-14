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

/* This file contains the implementation of the BLAS-3 function ssyrk */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

/* Use square 32x32 tiles to access and cache portions of source matrices A,B 
 * and result matrix C.
 */
#define TILE_DIM_LOG    (5)
#define TILE_DIM        (1 << TILE_DIM_LOG)
#define TILE_SIZE       (TILE_DIM*TILE_DIM)
#define SUP_TILE_DIM    (TILE_DIM*CUBLAS_SSYRK_GRIDW)

/* forward declaration, see ssyrk.cu for these */
__global__ void ssyrk_up_tr_main_hw (struct cublasSsyrkParams parms);
__global__ void ssyrk_up_nt_main_hw (struct cublasSsyrkParams parms);
__global__ void ssyrk_lo_tr_main_hw (struct cublasSsyrkParams parms);
__global__ void ssyrk_lo_nt_main_hw (struct cublasSsyrkParams parms);
__global__ void ssyrk_up_tr_main_sw (struct cublasSsyrkParams parms);
__global__ void ssyrk_up_nt_main_sw (struct cublasSsyrkParams parms);
__global__ void ssyrk_lo_tr_main_sw (struct cublasSsyrkParams parms);
__global__ void ssyrk_lo_nt_main_sw (struct cublasSsyrkParams parms);
__global__ void ssyrk_up_tr_main_hw_fulltile (struct cublasSsyrkParams parms);
__global__ void ssyrk_up_nt_main_hw_fulltile (struct cublasSsyrkParams parms);
__global__ void ssyrk_lo_tr_main_hw_fulltile (struct cublasSsyrkParams parms);
__global__ void ssyrk_lo_nt_main_hw_fulltile (struct cublasSsyrkParams parms);
__global__ void ssyrk_up_tr_main_sw_fulltile (struct cublasSsyrkParams parms);
__global__ void ssyrk_up_nt_main_sw_fulltile (struct cublasSsyrkParams parms);
__global__ void ssyrk_lo_tr_main_sw_fulltile (struct cublasSsyrkParams parms);
__global__ void ssyrk_lo_nt_main_sw_fulltile (struct cublasSsyrkParams parms);

__global__ void fast_ssyrk_up_tr_main_hw (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_up_nt_main_hw (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_lo_tr_main_hw (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_lo_nt_main_hw (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_up_tr_main_sw (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_up_nt_main_sw (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_lo_tr_main_sw (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_lo_nt_main_sw (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_up_tr_main_hw_fulltile (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_up_nt_main_hw_fulltile (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_lo_tr_main_hw_fulltile (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_lo_nt_main_hw_fulltile (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_up_tr_main_sw_fulltile (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_up_nt_main_sw_fulltile (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_lo_tr_main_sw_fulltile (struct cublasSsyrkParams parms);
__global__ void fast_ssyrk_lo_nt_main_sw_fulltile (struct cublasSsyrkParams parms);

typedef void (*pf) (struct cublasSsyrkParams parms);

static pf ssyrk_sw[16] = {
    ssyrk_lo_nt_main_sw,
    ssyrk_lo_tr_main_sw,
    ssyrk_up_nt_main_sw,
    ssyrk_up_tr_main_sw,
    fast_ssyrk_lo_nt_main_sw,
    fast_ssyrk_lo_tr_main_sw,
    fast_ssyrk_up_nt_main_sw,
    fast_ssyrk_up_tr_main_sw,
    ssyrk_lo_nt_main_sw_fulltile,
    ssyrk_lo_tr_main_sw_fulltile,
    ssyrk_up_nt_main_sw_fulltile,
    ssyrk_up_tr_main_sw_fulltile,
    fast_ssyrk_lo_nt_main_sw_fulltile,
    fast_ssyrk_lo_tr_main_sw_fulltile,
    fast_ssyrk_up_nt_main_sw_fulltile,
    fast_ssyrk_up_tr_main_sw_fulltile
};

static pf ssyrk_hw[16] = {
    ssyrk_lo_nt_main_hw,
    ssyrk_lo_tr_main_hw,
    ssyrk_up_nt_main_hw,
    ssyrk_up_tr_main_hw,
    fast_ssyrk_lo_nt_main_hw,
    fast_ssyrk_lo_tr_main_hw,
    fast_ssyrk_up_nt_main_hw,
    fast_ssyrk_up_tr_main_hw,
    ssyrk_lo_nt_main_hw_fulltile,
    ssyrk_lo_tr_main_hw_fulltile,
    ssyrk_up_nt_main_hw_fulltile,
    ssyrk_up_tr_main_hw_fulltile,
    fast_ssyrk_lo_nt_main_hw_fulltile,
    fast_ssyrk_lo_tr_main_hw_fulltile,
    fast_ssyrk_up_nt_main_hw_fulltile,
    fast_ssyrk_up_tr_main_hw_fulltile
};

/*
 * void 
 * cublasSsyr2k (char uplo, char trans, int n, int k, float alpha, 
 *               const float *A, int lda, const float *B, int ldb, 
 *               float beta, float *C, int ldc)
 *
 * performs one of the symmetric rank 2k operations
 * 
 *    C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, or 
 *    C = alpha * transpose(A) * B + alpha * transpose(B) * A + beta * C.
 *
 * Alpha and beta are single precision scalars. C is an n x n symmetric matrix
 * consisting of single precision elements and stored in either lower or upper 
 * storage mode. A and B are matrices consisting of single precision elements 
 * with dimension of n x k in the first case, and k x n in the second case.
 *
 * Input
 * -----
 * uplo   specifies whether the symmetric matrix C is stored in upper or lower
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper 
 *        triangular part of the symmetric matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the 
 *        lower triangular part of the symmetric matrix is to be references, 
 *        and the elements of the strictly upper triangular part are to be 
 *        infered from those in the lower triangular part.
 * trans  specifies the operation to be performed. If trans == 'N' or 'n', 
 *        C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, 
 *        If trans == 'T', 't', 'C', or 'c', C = alpha * transpose(A) * B + 
 *        alpha * transpose(B) * A + beta * C.
 * n      specifies the number of rows and the number columns of matrix C. If 
 *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If 
 *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A. 
 *        n must be at least zero.
 * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A. 
 *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of 
 *        matrix A. k must be at least zero.
 * alpha  single precision scalar multiplier.
 * A      single precision array of dimensions (lda, ka), where ka is k when 
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n', 
 *        the leading n x k part of array A must contain the matrix A, 
 *        otherwise the leading k x n part of the array must contain the matrix
 *        A.
 * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at 
 *        least max(1, n). Otherwise lda must be at least max(1,k).
 * B      single precision array of dimensions (lda, kb), where kb is k when 
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n', 
 *        the leading n x k part of array B must contain the matrix B, 
 *        otherwise the leading k x n part of the array must contain the matrix
 *        B.
 * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
 *        least max(1, n). Otherwise ldb must be at least max(1, k).
 * beta   single precision scalar multiplier applied to C. If beta is zero, C 
 *        does not have to be a valid input.
 * C      single precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
 *        the leading n x n triangular part of the array C must contain the 
 *        upper triangular part of the symmetric matrix C and the strictly 
 *        lower triangular part of C is not referenced. On exit, the upper 
 *        triangular part of C is overwritten by the upper trinagular part of 
 *        the updated matrix. If uplo == 'L' or 'l', the leading n x n 
 *        triangular part of the array C must contain the lower triangular part
 *        of the symmetric matrix C and the strictly upper triangular part of C
 *        is not referenced. On exit, the lower triangular part of C is 
 *        overwritten by the lower trinagular part of the updated matrix.
 * ldc    leading dimension of C. Must be at least max(1, n).
 *
 * Output
 * ------
 * C      updated according to alpha*A*transpose(B) + alpha*B*transpose(A) + 
 *        beta*C or alpha*transpose(A)*B + alpha*transpose(B)*A + beta*C
 *
 * Reference:   http://www.netlib.org/blas/ssyr2k.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSsyr2k (char uplo, char trans, int n, int k,
                                      float alpha, const float *A, int lda,
                                      const float *B, int ldb, float beta,
                                      float *C, int ldc)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasSsyrkParams params;
    cudaError_t cudaStat;
    int nrowa, upper, transpose;
    int info = 0;
    int usePureHwStepper;
    int useFastImul;
    int fullTilesOnly;
    int funcIdx;
    dim3 ctaDimsHw (((n+TILE_DIM-1)>>TILE_DIM_LOG),
                    ((n+TILE_DIM-1)>>TILE_DIM_LOG));
    dim3 ctaDimsSw (CUBLAS_SSYR2K_GRIDW, CUBLAS_SSYR2K_GRIDH);

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }

    upper = toupper (uplo) == 'U';
    transpose = (toupper (trans) == 'T') || (toupper (trans) == 'C');
    nrowa = transpose ? k : n;

    /* Test the input parameters. */

    info = 0;
    if ((!upper) && (!(toupper (uplo) == 'L'))) {
        info = 1;
    }
    else if ((!transpose) && (!(toupper (trans) == 'N'))) {
        info = 2;
    }
    else if (n < 0) {
        info = 3;
    }
    else if (k < 0) {
        info = 4;
    }
    else if (lda < imax (1, nrowa)) {
        info = 7;
    }
    else if (ldb < imax (1, nrowa)) {
        info = 9;
    }
    else if (ldc < imax (1, n)) {
        info = 12;
    }
    if (info) {
        cublasXerbla ("SSYR2K", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((n == 0) || (((alpha == 0.0f) || (k == 0)) && (beta == 1.0f))) {
        return;
    }

    memset (&params, 0, sizeof(params));
    params.upper = upper;
    params.transpose = transpose;
    params.n = n;
    params.k = k;
    params.alpha = alpha;
    params.A = A;
    params.lda = lda;
    params.B = B;
    params.ldb = ldb;
    params.beta = beta;
    params.C = C;
    params.ldc = ldc;

    /* we can eliminate checking for endcases if we know all tiles are fully
     * populated. Important benchmark case!
     */
    fullTilesOnly = (((n % TILE_DIM) == 0) &&
                     ((k % TILE_DIM) == 0));

    /* choose version using 24-bit multiplies if all dimensions are less than
     * 2001, so we can guarantee that no multiplication result exceeds (2000 *
     * 2000 * 4) < 2^24.
     */
    useFastImul =((params.lda <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                  (params.ldb <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                  (params.ldc <= CUBLAS_FASTIMUL_F_MAX_DIM) &&
                  (params.n   <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                  (params.k   <= CUBLAS_FASTIMUL_F_MAX_DIM));

    /* choose HW-only stepping if dimensions of result matrix are not exact 
     * multiples of a supertile, or exceed the maximum CTA grid dimensions.
     */
    usePureHwStepper = params.n < (CUBLAS_CTA_MAX_DIM * TILE_DIM);

    funcIdx = ((fullTilesOnly << 3) | (useFastImul << 2) | 
               (params.upper << 1) | params.transpose);

    cudaStat = cudaGetLastError(); /* clear error status */
    if (usePureHwStepper) {
        ssyrk_hw[funcIdx]<<<ctaDimsHw,CUBLAS_SSYR2K_THREAD_COUNT>>>(params);
    } else {
        ssyrk_sw[funcIdx]<<<ctaDimsSw,CUBLAS_SSYR2K_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* launch check */
    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }

    params.A = B;
    params.lda = ldb;
    params.B = A;
    params.ldb = lda;
    params.beta = 1.0f;

    cudaStat = cudaGetLastError(); /* clear error status */
    if (usePureHwStepper) {
        ssyrk_hw[funcIdx]<<<ctaDimsHw,CUBLAS_SSYR2K_THREAD_COUNT>>>(params);
    } else {
        ssyrk_sw[funcIdx]<<<ctaDimsSw,CUBLAS_SSYR2K_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* launch check */
    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}
