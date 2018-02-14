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
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"
#include "cublasP.h"

/* Use square 32x32 tiles to access and cache portions of source matrices A,B 
 * and result matrix C
 */
#define TILE_DIM_LOG    (5)
#define THREAD_COUNT    (CUBLAS_SGEMM_LARGE_THREAD_COUNT)
#include "sgemm_sizing.h"
#include "sgemm_common.h"

/*
 * void 
 * cublasSgemm (char transa, char transb, int m, int n, int k, float alpha, 
 *              const float *A, int lda, const float *B, int ldb, float beta, 
 *              float *C, int ldc)
 *
 * computes the product of matrix A and matrix B, multiplies the result 
 * by a scalar alpha, and adds the sum to the product of matrix C and
 * scalar beta. sgemm() performs one of the matrix-matrix operations:
 *
 *     C = alpha * op(A) * op(B) + beta * C,
 *
 * where op(X) is one of
 *
 *     op(X) = X   or   op(X) = transpose(X)
 *
 * alpha and beta are single precision scalars, and A, B and C are 
 * matrices consisting of single precision elements, with op(A) an m x k 
 * matrix, op(B) a k x n matrix, and C an m x n matrix. Matrices A, B, 
 * and C are stored in column major format, and lda, ldb, and ldc are
 * the leading dimensions of the two-dimensional arrays containing A, 
 * B, and C.
 *
 * Input
 * -----
 * transa specifies op(A). If transa = 'n' or 'N', op(A) = A. If 
 *        transa = 't', 'T', 'c', or 'C', op(A) = transpose(A)
 * transb specifies op(B). If transb = 'n' or 'N', op(B) = B. If 
 *        transb = 't', 'T', 'c', or 'C', op(B) = transpose(B)
 * m      number of rows of matrix op(A) and rows of matrix C
 * n      number of columns of matrix op(B) and number of columns of C
 * k      number of columns of matrix op(A) and number of rows of op(B) 
 * alpha  single precision scalar multiplier applied to op(A)op(B)
 * A      single precision array of dimensions (lda, k) if transa = 
 *        'n' or 'N'), and of dimensions (lda, m) otherwise. When transa =
 *        'N' or 'n' then lda must be at least  max( 1, m ), otherwise lda
 *        must be at least max(1, k).
 * lda    leading dimension of two-dimensional array used to store matrix A
 * B      single precision array of dimensions  (ldb, n) if transb =
 *        'n' or 'N'), and of dimensions (ldb, k) otherwise. When transb =
 *        'N' or 'n' then ldb must be at least  max (1, k), otherwise ldb
 *        must be at least max (1, n).
 * ldb    leading dimension of two-dimensional array used to store matrix B
 * beta   single precision scalar multiplier applied to C. If 0, C does
 *        not have to be a valid input
 * C      single precision array of dimensions (ldc, n). ldc must be at 
 *        least max (1, m).
 * ldc    leading dimension of two-dimensional array used to store matrix C
 *
 * Output
 * ------
 * C      updated based on C = alpha * op(A)*op(B) + beta * C
 *
 * Reference: http://www.netlib.org/blas/sgemm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if any of m, n, or k are < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSgemm (char transa, char transb, int m, int n,
                                     int k, float alpha, const float *A,
                                     int lda, const float *B, int ldb,
                                     float beta, float *C, int ldc)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    int ta, tb;
    int nrowa, nrowb;
    int info = 0;
    int useFastImul;

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
    }

    ta = ((toupper(transa) == 'T') || (toupper(transa) == 'C'));
    tb = ((toupper(transb) == 'T') || (toupper(transb) == 'C'));

    nrowa = ta ? k : m;
    nrowb = tb ? n : k;

    if ((toupper(transa) != 'N') && 
        (toupper(transa) != 'C') && 
        (toupper(transa) != 'T')) {
        info = 1;
    } 
    else if ((toupper(transb) != 'N') && 
             (toupper(transb) != 'C') && 
             (toupper(transb) != 'T')) {
        info = 2;
    }
    else if (m < 0) {
        info = 3;
    }
    else if (n < 0) {
        info = 4;
    }
    else if (k < 0) {
        info = 5;
    }
    else if (lda < imax(1, nrowa)) {
        info = 8;
    }
    else if (ldb < imax(1, nrowb)) {
        info = 10;
    }
    else if (ldc < imax(1, m)) {
        info = 13;
    }
    if (info) {
        cublasXerbla ("SGEMM ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((m == 0) || (n == 0) || 
        (((alpha == 0.0f) || (k == 0)) && (beta == 1.0f))) {
        return;
    }

    /* if the matrices are all small, it's better to use smaller tiles where
     * each thread handles only a single result matrix element. This brings
     * more threads to bear on a problem of a given size, compared to the
     * standard mathod, and having more threads in play increases performance.
     * The cutover value has been determined experimentally.
     */
    if (((m * n) <= CUBLAS_SMALL_SGEMM_MAT_MAX_ELEMS) &&
        ((m * k) <= CUBLAS_SMALL_SGEMM_MAT_MAX_ELEMS) &&
        ((n * k) <= CUBLAS_SMALL_SGEMM_MAT_MAX_ELEMS)) {
        cublasSmallSgemm (ctx, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                          beta, C, ldc);
        return;
    }

    /* choose version using 24-bit multiplies if all dimensions are less than
     * 2001, so we can guarantee that no multiplication result exceeds (2000 *
     * 2000 * 4) < 2^24.
     */
    useFastImul =((lda <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                  (ldb <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                  (ldc <= CUBLAS_FASTIMUL_F_MAX_DIM) &&
                  (m   <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                  (n   <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                  (k   <= CUBLAS_FASTIMUL_F_MAX_DIM));

    if (useFastImul) {
        cublasFastSgemm (ctx, transa, transb, m, n, k, alpha, A, lda, B, ldb, 
                         beta, C, ldc);
        return;
    }        

    cublasLargeSgemm (ctx, transa, transb, m, n, k, alpha, A, lda, B, ldb, 
                      beta, C, ldc);
}
