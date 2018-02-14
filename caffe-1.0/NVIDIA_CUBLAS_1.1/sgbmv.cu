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

/* This file contains the implementation of the BLAS-2 function sgbmv */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void sgbmvn_main (struct cublasSgbmvParams parms);
__global__ void sgbmvt_main (struct cublasSgbmvParams parms);

/*
 * void 
 * cublasSgbmv (char trans, int m, int n, int kl, int ku, float alpha,
 *              const float *A, int lda, const float *x, int incx, float beta,
 *              float *y, int incy);
 *
 * performs one of the matrix-vector operations
 *
 *    y = alpha*op(A)*x + beta*y,  op(A)=A or op(A) = transpose(A)
 *
 * alpha and beta are single precision scalars. x and y are single precision
 * vectors. A is an m by n band matrix consisting of single precision elements
 * with kl sub-diagonals and ku super-diagonals.
 *
 * Input
 * -----
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T', 
 *        't', 'C', or 'c', op(A) = transpose(A)
 * m      specifies the number of rows of the matrix A. m must be at least 
 *        zero.
 * n      specifies the number of columns of the matrix A. n must be at least
 *        zero.
 * kl     specifies the number of sub-diagonals of matrix A. It must be at 
 *        least zero.
 * ku     specifies the number of super-diagonals of matrix A. It must be at
 *        least zero.
 * alpha  single precision scalar multiplier applied to op(A).
 * A      single precision array of dimensions (lda, n). The leading
 *        (kl + ku + 1) x n part of the array A must contain the band matrix A,
 *        supplied column by column, with the leading diagonal of the matrix 
 *        in row (ku + 1) of the array, the first super-diagonal starting at 
 *        position 2 in row ku, the first sub-diagonal starting at position 1
 *        in row (ku + 2), and so on. Elements in the array A that do not 
 *        correspond to elements in the band matrix (such as the top left 
 *        ku x ku triangle) are not referenced.
 * lda    leading dimension of A. lda must be at least (kl + ku + 1).
 * x      single precision array of length at least (1+(n-1)*abs(incx)) when 
 *        trans == 'N' or 'n' and at least (1+(m-1)*abs(incx)) otherwise.
 * incx   specifies the increment for the elements of x. incx must not be zero.
 * beta   single precision scalar multiplier applied to vector y. If beta is 
 *        zero, y is not read.
 * y      single precision array of length at least (1+(m-1)*abs(incy)) when 
 *        trans == 'N' or 'n' and at least (1+(n-1)*abs(incy)) otherwise. If 
 *        beta is zero, y is not read.
 * incy   On entry, incy specifies the increment for the elements of y. incy 
 *        must not be zero.
 *
 * Output
 * ------
 * y      updated according to y = alpha*op(A)*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/sgbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSgbmv (char trans, int m, int n, int kl, int ku, 
                                     float alpha, const float *A, int lda, 
                                     const float *x, int incx, float beta, 
                                     float *y, int incy)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasSgbmvParams params;
    cudaError_t cudaStat;
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
    else if (kl < 0) {
        info = 4;
    }
    else if (ku < 0) {
        info = 5;
    }
    else if (lda < (kl + ku + 1)) {
        info = 8;
    }
    else if (incx == 0) {
        info = 10;
    }
    else if (incy == 0) {
        info = 13;
    }
    if (info) {
        cublasXerbla ("SGBMV ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((m == 0) || (n == 0) || ((alpha == 0.0f) && (beta == 1.0f))) {
        return;
    }

    memset (&params, 0, sizeof(params));
    params.trans = toupper(trans) != 'N';
    params.m = m;
    params.n = n;
    params.kl = kl;
    params.ku = ku;
    params.alpha = alpha;
    params.A = A;
    params.lda = lda;
    params.x = x;
    params.incx = incx;
    params.beta = beta;
    params.y = y;
    params.incy = incy;

    cudaStat = cudaGetLastError(); /* clear error status */
    if (params.trans) {
        sgbmvt_main<<<CUBLAS_SGBMV_CTAS,CUBLAS_SGBMV_THREAD_COUNT>>>(params);
    } else {
        sgbmvn_main<<<CUBLAS_SGBMV_CTAS,CUBLAS_SGBMV_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

#define IDXA(row,col)       ((parms.lda*(col))+((row)-(col)+(parms.ku)))
#define IDXX(i)             (startx + ((i) * parms.incx))
#define IDXY(i)             (starty + ((i) * parms.incy))

#define X_ELEMS_PER_THREAD  (4)
#define IINC                (CUBLAS_SGBMV_CTAS * CUBLAS_SGBMV_THREAD_COUNT)
#define JINC                (CUBLAS_SGBMV_THREAD_COUNT * X_ELEMS_PER_THREAD)
#define XINC                (CUBLAS_SGBMV_THREAD_COUNT)

__shared__ float XX[JINC];  /* cached portion of vector x */

__global__ void sgbmvn_main (struct cublasSgbmvParams parms) 
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
    starty = (parms.incy >= 0) ? 0 : ((1 - parms.m) * parms.incy);
    for (i = 0; i < parms.m; i += IINC) {
        /* first row being processed by this CTA */
        ii = i + blockIdx.x * CUBLAS_SGBMV_THREAD_COUNT;
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
                int jjStart = ii - parms.kl; /* may be negative */
                int jjEnd   = ii + parms.ku + 1;
                if ((jjEnd > j) && (jjStart < jjLimit)) {
                    jj = max (j, jjStart);
//                        printf ("row %d initial j=%d less than %d\n", ii, jj, (min (jjLimit, jjEnd)));fflush(stdout);
                    while (jj < (min (jjLimit, jjEnd))) {
//                            printf ("%d,%d A[%d]=%g\n",ii,jj,IDXA(ii,jj),parms.A[IDXA(ii,jj)]);fflush(stdout);
                        sdot += parms.A[IDXA(ii,jj)] * XX[jj-j];
                        jj++;
                    }
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

__global__ void sgbmvt_main (struct cublasSgbmvParams parms) 
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
    startx = (parms.incx >= 0) ? 0 : ((1 - parms.m) * parms.incx);
    starty = (parms.incy >= 0) ? 0 : ((1 - parms.n) * parms.incy);
    for (i = 0; i < parms.n; i += IINC) {
        /* first row being processed by this CTA */
        ii = i + blockIdx.x * CUBLAS_SGBMV_THREAD_COUNT;
        if (ii >= parms.n) break; /* nothing to do for this CTA */
        ii += tid; /* row being processed by this thread */
        sdot = 0.0f; /* initialize dot product handled by this thread */
        /* iterate over chunks of rows. These chunks are very large, so
         * in many case we'll only executed the loop body once, i.e. we'll
         * process the whole row in one fell swoop.
         */
        for (j = 0; j < parms.m; j += JINC) {
            int jjLimit = min (j + JINC, parms.m);
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
//                printf ("XX[%d]=%g\n",tid+0*XINC,XX[tid+0*XINC]);fflush(stdout);
            }
#else
#error current code cannot handle X_ELEMS_PER_THREAD != 4
#endif
            __syncthreads ();
            
            if (ii < parms.n) { /* if this row is active, accumulate dp */
                int jjStart = ii - parms.ku; /* may be negative */
                int jjEnd   = ii + parms.kl + 1;
                if ((jjEnd > j) && (jjStart < jjLimit)) {
                    jj = max (j, jjStart);
//                    printf ("col %d initial j=%d less than %d\n", ii, jj, (min (jjLimit, jjEnd)));fflush(stdout);
                    while (jj < (min (jjLimit, jjEnd))) {
//                        printf ("%d,%d A[%d]=%g\n",jj,ii,IDXA(jj,ii),parms.A[IDXA(jj,ii)]);fflush(stdout);
                        sdot += parms.A[IDXA(jj,ii)] * XX[jj-j];
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


