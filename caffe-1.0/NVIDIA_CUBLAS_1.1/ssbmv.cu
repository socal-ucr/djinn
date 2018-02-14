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

/* This file contains the implementation of the BLAS-2 function ssbmv */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void ssbmvu_main (struct cublasSsbmvParams parms);
__global__ void ssbmvl_main (struct cublasSsbmvParams parms);

/*
 * void 
 * cublasSsbmv (char uplo, int n, int k, float alpha, const float *A, int lda,
 *              const float *x, int incx, float beta, float *y, int incy)
 *
 * performs the matrix-vector operation
 *
 *     y := alpha*A*x + beta*y
 *
 * alpha and beta are single precision scalars. x and y are single precision
 * vectors with n elements. A is an n by n symmetric band matrix consisting 
 * of single precision elements, with k super-diagonals and the same number
 * of subdiagonals.
 *
 * Input
 * -----
 * uplo   specifies whether the upper or lower triangular part of the symmetric
 *        band matrix A is being supplied. If uplo == 'U' or 'u', the upper 
 *        triangular part is being supplied. If uplo == 'L' or 'l', the lower 
 *        triangular part is being supplied.
 * n      specifies the number of rows and the number of columns of the
 *        symmetric matrix A. n must be at least zero.
 * k      specifies the number of super-diagonals of matrix A. Since the matrix
 *        is symmetric, this is also the number of sub-diagonals. k must be at
 *        least zero.
 * alpha  single precision scalar multiplier applied to A*x.
 * A      single precision array of dimensions (lda, n). When uplo == 'U' or 
 *        'u', the leading (k + 1) x n part of array A must contain the upper
 *        triangular band of the symmetric matrix, supplied column by column,
 *        with the leading diagonal of the matrix in row (k+1) of the array,
 *        the first super-diagonal starting at position 2 in row k, and so on.
 *        The top left k x k triangle of the array A is not referenced. When
 *        uplo == 'L' or 'l', the leading (k + 1) x n part of the array A must
 *        contain the lower triangular band part of the symmetric matrix, 
 *        supplied column by column, with the leading diagonal of the matrix in
 *        row 1 of the array, the first sub-diagonal starting at position 1 in
 *        row 2, and so on. The bottom right k x k triangle of the array A is
 *        not referenced.
 * lda    leading dimension of A. lda must be at least (k + 1).
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   single precision scalar multiplier applied to vector y. If beta is 
 *        zero, y is not read.
 * y      single precision array of length at least (1 + (n - 1) * abs(incy)). 
 *        If beta is zero, y is not read.
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to alpha*A*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/ssbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_INVALID_VALUE    if k or n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSsbmv (char uplo, int n, int k, float alpha, 
                                     const float *A, int lda, const float *x, 
                                     int incx, float beta, float *y, int incy)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasSsbmvParams params;
    cudaError_t cudaStat;
    int info = 0;

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }

    /* check inputs */
    if ((toupper (uplo) != 'U') &&
        (toupper (uplo) != 'L')) {
        info = 1;
    } 
    else if (n < 0) {
        info = 2;
    }
    else if (k < 0) {
        info = 3;
    }
    else if (lda < (k + 1)) {
        info = 6;
    }
    else if (incx == 0) {
        info = 8;
    }
    else if (incy == 0) {
        info = 11;
    }
    if (info) {
        cublasXerbla ("SSBMV ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((n == 0) || ((alpha == 0.0f) && (beta == 1.0f))) {
        return;
    }

    memset (&params, 0, sizeof(params));
    params.up = toupper(uplo) == 'U';
    params.n = n;
    params.k = k;
    params.alpha = alpha;
    params.A = A;
    params.lda = lda;
    params.x = x;
    params.incx = incx;
    params.beta = beta;
    params.y = y;
    params.incy = incy;

    cudaStat = cudaGetLastError(); /* clear error status */
    if (params.up) {
        ssbmvu_main<<<CUBLAS_SSBMV_CTAS,CUBLAS_SSBMV_THREAD_COUNT>>>(params);
    } else {
        ssbmvl_main<<<CUBLAS_SSBMV_CTAS,CUBLAS_SSBMV_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

#define IDXA_UP(row,col)    ((parms.lda*(col))+(parms.k)+((row)-(col)))
#define IDXA_LO(row,col)    ((parms.lda*(col))+((row)-(col)))
#define IDXX(i)             (startx + ((i) * parms.incx))
#define IDXY(i)             (starty + ((i) * parms.incy))

#define X_ELEMS_PER_THREAD  (4)
#define IINC                (CUBLAS_SSBMV_CTAS * CUBLAS_SSBMV_THREAD_COUNT)
#define JINC                (CUBLAS_SSBMV_THREAD_COUNT * X_ELEMS_PER_THREAD)
#define XINC                (CUBLAS_SSBMV_THREAD_COUNT)

__shared__ float XX[JINC];  /* cached portion of vector x */

__global__ void ssbmvu_main (struct cublasSsbmvParams parms) 
{
    int i, ii, j, jj, idx, incr, tid;
    float sdot;
    int startx;
    int starty;

    /*
     * NOTE: wrapper must ensure that parms.n >= 0, and that parms.incx and 
     *       parms.incy are != 0 
     */

    tid = threadIdx.x;
    startx = (parms.incx >= 0) ? 0 : ((1 - parms.n) * parms.incx);
    starty = (parms.incy >= 0) ? 0 : ((1 - parms.n) * parms.incy);
    for (i = 0; i < parms.n; i += IINC) {
        /* first row being processed by this CTA */
        ii = i + blockIdx.x * CUBLAS_SSBMV_THREAD_COUNT;
        if (ii >= parms.n) break; /* nothing to do for this CTA */
        ii += tid; /* row being processed by this thread */
        sdot = 0.0f; /* initialize dot product handled by this thread */
        /* iterate over chunks of rows. These chunks are very large, so
         * in many case we'll only executed the loop body once, i.e. we'll
         * process the whole row in one fell swoop.
         */
        for (j = 0; j < parms.n; j += JINC) {
            int jjLimit = min (j + JINC, parms.n);
            incr = XINC * parms.incx;
            jj = j + tid;

            __syncthreads ();
            idx = IDXX(jj);
#if (X_ELEMS_PER_THREAD == 4)
            if (jj < (jjLimit - 3 * XINC)) {
                XX[tid+0*XINC] = parms.alpha * parms.x[idx + 0 * incr];
                XX[tid+1*XINC] = parms.alpha * parms.x[idx + 1 * incr];
                XX[tid+2*XINC] = parms.alpha * parms.x[idx + 2 * incr];
                XX[tid+3*XINC] = parms.alpha * parms.x[idx + 3 * incr];
            }
            else if (jj < (jjLimit - 2 * XINC)) {
                XX[tid+0*XINC] = parms.alpha * parms.x[idx + 0 * incr];
                XX[tid+1*XINC] = parms.alpha * parms.x[idx + 1 * incr];
                XX[tid+2*XINC] = parms.alpha * parms.x[idx + 2 * incr];
            }
            else if (jj < (jjLimit - 1 * XINC)) {
                XX[tid+0*XINC] = parms.alpha * parms.x[idx + 0 * incr];
                XX[tid+1*XINC] = parms.alpha * parms.x[idx + 1 * incr];
            }
            else if (jj < (jjLimit - 0 * XINC)) {
                XX[tid+0*XINC] = parms.alpha * parms.x[idx + 0 * incr];
            }
#else
#error current code cannot handle X_ELEMS_PER_THREAD != 4
#endif
            __syncthreads ();
            
            if (ii < parms.n) { /* if this row is active, accumulate dp */
                int jjStart = ii - parms.k; /* may be negative */
                int jjEnd   = ii + parms.k + 1;
                if ((jjEnd > j) && (jjStart < jjLimit)) {
                    jj = max (j, jjStart);
                    while (jj < (min (jjLimit, jjEnd))) {
                        int idx = (ii < jj) ? IDXA_UP(ii,jj) : IDXA_UP(jj,ii);
                        sdot += parms.A[idx] * XX[jj-j];
                        jj++;
                    }
                }
            }
        }
        if (ii < parms.n) { /* if this row is active, write out dp */
            idx = IDXY(ii);
            if (parms.beta != 0.0f) {
                sdot += parms.beta * parms.y[idx];
            }
            parms.y[idx] = sdot;
        }
    }
}

__global__ void ssbmvl_main (struct cublasSsbmvParams parms) 
{
    int i, ii, j, jj, idx, incr, tid;
    float sdot;
    int startx;
    int starty;

    /*
     * NOTE: wrapper must ensure that parms.n >= 0, and that parms.incx and 
     *       parms.incy are != 0 
     */

    tid = threadIdx.x;
    startx = (parms.incx >= 0) ? 0 : ((1 - parms.n) * parms.incx);
    starty = (parms.incy >= 0) ? 0 : ((1 - parms.n) * parms.incy);
    for (i = 0; i < parms.n; i += IINC) {
        /* first row being processed by this CTA */
        ii = i + blockIdx.x * CUBLAS_SSBMV_THREAD_COUNT;
        if (ii >= parms.n) break; /* nothing to do for this CTA */
        ii += tid; /* row being processed by this thread */
        sdot = 0.0f; /* initialize dot product handled by this thread */
        /* iterate over chunks of rows. These chunks are very large, so
         * in many case we'll only executed the loop body once, i.e. we'll
         * process the whole row in one fell swoop.
         */
        for (j = 0; j < parms.n; j += JINC) {
            int jjLimit = min (j + JINC, parms.n);
            incr = XINC * parms.incx;
            jj = j + tid;
            __syncthreads ();
            idx = IDXX(jj);
#if (X_ELEMS_PER_THREAD == 4)
            if (jj < (jjLimit - 3 * XINC)) {
                XX[tid+0*XINC] = parms.alpha * parms.x[idx + 0 * incr];
                XX[tid+1*XINC] = parms.alpha * parms.x[idx + 1 * incr];
                XX[tid+2*XINC] = parms.alpha * parms.x[idx + 2 * incr];
                XX[tid+3*XINC] = parms.alpha * parms.x[idx + 3 * incr];
            }
            else if (jj < (jjLimit - 2 * XINC)) {
                XX[tid+0*XINC] = parms.alpha * parms.x[idx + 0 * incr];
                XX[tid+1*XINC] = parms.alpha * parms.x[idx + 1 * incr];
                XX[tid+2*XINC] = parms.alpha * parms.x[idx + 2 * incr];
            }
            else if (jj < (jjLimit - 1 * XINC)) {
                XX[tid+0*XINC] = parms.alpha * parms.x[idx + 0 * incr];
                XX[tid+1*XINC] = parms.alpha * parms.x[idx + 1 * incr];
            }
            else if (jj < (jjLimit - 0 * XINC)) {
                XX[tid+0*XINC] = parms.alpha * parms.x[idx + 0 * incr];
            }
#else
#error current code cannot handle X_ELEMS_PER_THREAD != 4
#endif
            __syncthreads ();
            
            if (ii < parms.n) { /* if this row is active, accumulate dp */
                int jjStart = ii - parms.k; /* may be negative */
                int jjEnd   = ii + parms.k + 1;
                if ((jjEnd > j) && (jjStart < jjLimit)) {
                    jj = max (j, jjStart);
                    while (jj < (min (jjLimit, jjEnd))) {
                        int idx = (ii > jj) ? IDXA_LO(ii,jj) : IDXA_LO(jj,ii);
                        sdot += parms.A[idx] * XX[jj-j];
                        jj++;
                    }
                }
            }
        }
        if (ii < parms.n) { /* if this row is active, write out dp */
            idx = IDXY(ii);
            if (parms.beta != 0.0f) {
                sdot += parms.beta * parms.y[idx];
            }
            parms.y[idx] = sdot;
        }
    }
}


