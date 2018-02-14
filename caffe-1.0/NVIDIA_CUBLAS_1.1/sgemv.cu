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

/* This file contains the implementation of the BLAS-2 function sgemv */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void sgemvt_main (struct cublasSgemvParams parms);
__global__ void sgemvn_main (struct cublasSgemvParams parms);

/*
 * cublasSgemv (char trans, int m, int n, float alpha, const float *A, 
 *              int lda, const float *x, int incx, float beta, float *y, 
 *              int incy)
 *
 * performs one of the matrix-vector operations
 *
 *    y = alpha * op(A) * x + beta * y,
 *
 * where op(A) is one of
 *
 *    op(A) = A   or   op(A) = transpose(A)
 *
 * where alpha and beta are single precision scalars, x and y are single 
 * precision vectors, and A is an m x n matrix consisting of single precision
 * elements. Matrix A is stored in column major format, and lda is the leading
 * dimension of the two-dimensional array in which A is stored.
 *
 * Input
 * -----
 * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If trans =
 *        trans = 't', 'T', 'c', or 'C', op(A) = transpose(A)
 * m      specifies the number of rows of the matrix A. m must be at least 
 *        zero.
 * n      specifies the number of columns of the matrix A. n must be at least 
 *        zero.
 * alpha  single precision scalar multiplier applied to op(A).
 * A      single precision array of dimensions (lda, n) if trans = 'n' or 
 *        'N'), and of dimensions (lda, m) otherwise. lda must be at least 
 *        max(1, m) and at least max(1, n) otherwise.
 * lda    leading dimension of two-dimensional array used to store matrix A
 * x      single precision array of length at least (1 + (n - 1) * abs(incx))
 *        when trans = 'N' or 'n' and at least (1 + (m - 1) * abs(incx)) 
 *        otherwise.
 * incx   specifies the storage spacing between elements of x. incx must not 
 *        be zero.
 * beta   single precision scalar multiplier applied to vector y. If beta 
 *        is zero, y is not read.
 * y      single precision array of length at least (1 + (m - 1) * abs(incy))
 *        when trans = 'N' or 'n' and at least (1 + (n - 1) * abs(incy)) 
 *        otherwise.
 * incy   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 *
 * Output
 * ------
 * y      updated according to alpha * op(A) * x + beta * y
 *
 * Reference: http://www.netlib.org/blas/sgemv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0, or if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSgemv (char trans, int m, int n, float alpha, 
                                     const float *A, int lda, const float *x, 
                                     int incx, float beta, float *y, int incy)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasSgemvParams params;
    cudaError_t cudaStat;
    int transpose;
    int info = 0;

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }

    /* check inputs */
    if ((toupper (trans) != 'N') && 
        (toupper (trans) != 'T') && 
        (toupper (trans) != 'C')) {
        info = 1;
    } 
    else if (m < 0) {
        info = 2;
    }
    else if (n < 0) {
        info = 3;
    }
    else if (lda < imax (1, m)) {
        info = 6;
    }
    else if (incx == 0) {
        info = 8;
    }
    else if (incy == 0) {
        info = 11;
    }
    if (info) {
        cublasXerbla ("SGEMV ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((m == 0) || (n == 0) || ((beta == 1.0f) && (alpha == 0.0f))) {
        return;
    }

    memset (&params, 0, sizeof(params));
    transpose = ((toupper(trans) == 'T') || (toupper(trans) == 'C'));

    params.m = m;
    params.n = n;
    params.alpha = alpha;
    params.A = A;
    params.lda = lda;
    params.x = x;
    params.incx = incx;
    params.beta = beta;
    params.y = y;
    params.incy = incy;

    cudaStat = cudaGetLastError(); /* clear error status */
    if (transpose) {
        sgemvt_main<<<CUBLAS_SGEMVT_CTAS,CUBLAS_SGEMVT_THREAD_COUNT>>>(params);
    } else {
        sgemvn_main<<<CUBLAS_SGEMVN_CTAS,CUBLAS_SGEMVN_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

/* dimension m, counter i */
/* dimension n, counter j */

/* column-major ordering */
#define IDXA(row,col)       (parms.lda*(col)+(row))
#define IDXX(i)             (startx + ((i) * parms.incx))
#define IDXY(i)             (starty + ((i) * parms.incy))

#define THREAD_CNT          (CUBLAS_SGEMVT_THREAD_COUNT)
#define TILEW_LOG           (5)
#define TILEW               (1 << TILEW_LOG)
#define TILEH_LOG           (5)
#define TILEH               (1 << TILEH_LOG)
#define TILE_ELEMS          ((TILEW) * (TILEH))
#define TJINC               ((CUBLAS_SGEMVT_CTAS) * (TILEW))
#define TIINC               (TILEH)

#if ((TILE_ELEMS) % THREAD_CNT != 0)
#error number of elements in tile must be divisible by thread count
#endif
#define A_ELEMS_PER_THREAD  ((TILE_ELEMS) / (THREAD_CNT))
#define SLICEW              ((TILEW) / (A_ELEMS_PER_THREAD))

__shared__ float XX[TILEH];             /* cached portion of vector x */
__shared__ float AA[(TILEH+1)*TILEW];   /* cached portion of matrix A */

__global__ void sgemvt_main (struct cublasSgemvParams parms) 
{
    int i, ii, j, jj, jjjLimit, iiLimit, idx, jjj, x, startx, starty, tid;
    float sdot;

    /*
     * NOTE: wrapper must ensure that parms.m and parms.n are >= 0, and
     *       that parms.incx and parms.incy are != 0 
     */
    
    tid = threadIdx.x;
    startx = (parms.incx >= 0) ? 0 : ((1 - parms.m) * parms.incx);
    starty = (parms.incy >= 0) ? 0 : ((1 - parms.n) * parms.incy);
    /* step CTA array over the rows */
    for (j = 0; j < parms.n; j += TJINC) {
        /* first column being processed by this CTA */
        jj = j + blockIdx.x * TILEW;
        if (jj >= parms.n) break; /* nothing to do for this CTA */
        jjjLimit = min (jj + TILEW, parms.n) - jj;

        sdot = 0.0f;

        /* process rows in chunks of TILEH */
        for (i = 0; i < parms.m; i += TILEH) {
            iiLimit = min (i + TILEH, parms.m) - i;
            __syncthreads ();
            /* copy a 32 element chunk of vector x */
            if (tid < iiLimit) {
                XX[tid] = parms.x[IDXX(i + tid)];
            }
            /* copy a 32x32 element tile of matrix, with 8 elements copied
             * by each of the 128 threads
             */
            jjj = tid >> TILEH_LOG;
            ii  = tid & (TILEH - 1);
            idx = jjj * (TILEH + 1) + ii;
            x   = IDXA(i+ii, jj+jjj);
            while (jjj < jjjLimit) {
                if (ii < iiLimit) {
                    AA[idx] = parms.alpha * parms.A[x];
                }
                jjj += SLICEW; 
                idx += SLICEW * (TILEH + 1);
                x   += SLICEW * parms.lda;
            }
            __syncthreads ();
            /* if this column is active, accumulate dot product */
            if (tid < jjjLimit) {
                ii = 0;
                /* each thread handles one column */
                idx = tid * (TILEH + 1);
                while (ii < (iiLimit - 15)) {
                    sdot += AA[idx + 0] * XX[ii + 0];
                    sdot += AA[idx + 1] * XX[ii + 1];
                    sdot += AA[idx + 2] * XX[ii + 2];
                    sdot += AA[idx + 3] * XX[ii + 3];
                    sdot += AA[idx + 4] * XX[ii + 4];
                    sdot += AA[idx + 5] * XX[ii + 5];
                    sdot += AA[idx + 6] * XX[ii + 6];
                    sdot += AA[idx + 7] * XX[ii + 7];
                    sdot += AA[idx + 8] * XX[ii + 8];
                    sdot += AA[idx + 9] * XX[ii + 9];
                    sdot += AA[idx +10] * XX[ii +10];
                    sdot += AA[idx +11] * XX[ii +11];
                    sdot += AA[idx +12] * XX[ii +12];
                    sdot += AA[idx +13] * XX[ii +13];
                    sdot += AA[idx +14] * XX[ii +14];
                    sdot += AA[idx +15] * XX[ii +15];
                    idx  += 16;
                    ii   += 16;
                }
                while (ii < (iiLimit - 0)) {
                    sdot += AA[idx++] * XX[ii++];
                }
            }
        }
        /* if this row is active, update result element */
        if (tid < jjjLimit) {
            jjj = jj + tid; /* row being processed by this thread */
            idx = IDXY(jjj);
            if (parms.beta != 0.0f) {
                sdot += parms.beta * parms.y[idx];
            }
            parms.y[idx] = sdot;
        }
    }
}

#define X_ELEMS_PER_THREAD  (4)
#define IINC                (CUBLAS_SGEMVN_CTAS * CUBLAS_SGEMVN_THREAD_COUNT)
#define JINC                (CUBLAS_SGEMVN_THREAD_COUNT * X_ELEMS_PER_THREAD)
#define XINC                (CUBLAS_SGEMVN_THREAD_COUNT)

__global__ void sgemvn_main (struct cublasSgemvParams parms) 
{
    __shared__ float XX[JINC];
    int i, ii, j, jj, idx, incr, tid;
    float sdot;
    int startx;
    int starty;

    /*
     * NOTE: wrapper must ensure that parms.m and parms.n are >= 0, and
     *       that parms.incx and parms.incy are > 0 
     */

    tid = threadIdx.x;
    startx = (parms.incx >= 0) ? 0 : ((1 - parms.n) * parms.incx);
    starty = (parms.incy >= 0) ? 0 : ((1 - parms.m) * parms.incy);
    /* step CTA array over the rows */
    for (i = 0; i < parms.m; i += IINC) {
        /* first row being processed by this CTA */
        ii = i + blockIdx.x * CUBLAS_SGEMVN_THREAD_COUNT;            
        if (ii >= parms.m) break; /* nothing to do for this CTA */
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
            
            if (ii < parms.m) { /* if this row is active, accumulate dp */
                idx = IDXA(ii, j);
                incr = parms.lda;
                jjLimit = jjLimit - j;
                jj = 0;
                while (jj < (jjLimit - 5)) {
                    sdot += parms.A[idx + 0*incr] * XX[jj+ 0];
                    sdot += parms.A[idx + 1*incr] * XX[jj+ 1];
                    sdot += parms.A[idx + 2*incr] * XX[jj+ 2];
                    sdot += parms.A[idx + 3*incr] * XX[jj+ 3];
                    sdot += parms.A[idx + 4*incr] * XX[jj+ 4];
                    sdot += parms.A[idx + 5*incr] * XX[jj+ 5];
                    jj   += 6;
                    idx  += 6 * incr;
                }
                while (jj < jjLimit) {
                    sdot += parms.A[idx + 0*incr] * XX[jj+ 0];
                    jj   += 1;
                    idx  += 1 * incr;
                }
            }
        }
        if (ii < parms.m) { /* if this row is active, write out dp */
            idx = IDXY(ii);
            if (parms.beta != 0.0f) {
                sdot += parms.beta * parms.y[idx];
            }
           parms.y[idx] = sdot;
        }
    }
}
