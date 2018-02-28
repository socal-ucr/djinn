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

/* This file contains the implementation of the BLAS-2 function sspmv */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "caffe/cublas-1.1/cublas11.h"   /* CUBLAS public header file  */
#include "caffe/cublas-1.1/cublasP.h"  /* CUBLAS private header file */

__global__ void sspmv_up_main (struct cublasSspmvParams parms);
__global__ void sspmv_lo_main (struct cublasSspmvParams parms);

/*
 * void 
 * cublasSspmv (char uplo, int n, float alpha, const float *AP, const float *x,
 *              int incx, float beta, float *y, int incy)
 *
 * performs the matrix-vector operation
 * 
 *    y = alpha * A * x + beta * y
 *
 * Alpha and beta are single precision scalars, and x and y are single 
 * precision vectors with n elements. A is a symmetric n x n matrix 
 * consisting of single precision elements that is supplied in packed form.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array AP. If uplo == 'U' or 'u', then the upper 
 *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then 
 *        the lower triangular part of A is supplied in AP.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  single precision scalar multiplier applied to A*x.
 * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part 
 *        of the symmetric matrix A, packed sequentially, column by column; 
 *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If 
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part 
 *        of the symmetric matrix A, packed sequentially, column by column; 
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   single precision scalar multiplier applied to vector y;
 * y      single precision array of length at least (1 + (n - 1) * abs(incy)). 
 *        If beta is zero, y is not read. 
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to y = alpha*A*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/sspmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSspmv (char uplo, int n, float alpha,
                                     const float *AP, const float *x,
                                     int incx, float beta, float *y, int incy)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasSspmvParams params;
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
    else if (incx == 0) {
        info = 6;
    }
    else if (incy == 0) {
        info = 9;
    }
    if (info) {
        cublasXerbla ("SSPMV ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((n == 0) || ((beta == 1.0f) && (alpha == 0.0f))) {
        return;
    }

    memset (&params, 0, sizeof(params));
    params.up = toupper(uplo) == 'U';
    params.n = n;
    params.alpha = alpha;
    params.AP = AP;
    params.x = x;
    params.incx = incx;
    params.beta = beta;
    params.y = y;
    params.incy = incy;

    cudaStat = cudaGetLastError(); /* clear error status */
    if (params.up) {
        sspmv_up_main<<<CUBLAS_SSPMV_CTAS,CUBLAS_SSPMV_THREAD_COUNT>>>(params);
    } else {
        sspmv_lo_main<<<CUBLAS_SSPMV_CTAS,CUBLAS_SSPMV_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

/* dimension m, counter i */
/* dimension n, counter j */

/* column-major ordering */
#define IDXX(i)             (startx + ((i) * parms.incx))
#define IDXY(i)             (starty + ((i) * parms.incy))

#define X_ELEMS_PER_THREAD  (4)
#define IINC                (CUBLAS_SSPMV_CTAS * CUBLAS_SSPMV_THREAD_COUNT)
#define JINC                (CUBLAS_SSPMV_THREAD_COUNT * X_ELEMS_PER_THREAD)
#define XINC                (CUBLAS_SSPMV_THREAD_COUNT)

__shared__ float XX[JINC];  /* cached portion of vector x */

__global__ void sspmv_up_main (struct cublasSspmvParams parms) 
{
#undef  UP
#define UP 1
#include "caffe/cublas-1.1/sspmv.h"
}

__global__ void sspmv_lo_main (struct cublasSspmvParams parms) 
{
#undef  UP
#define UP 0
#include "caffe/cublas-1.1/sspmv.h"
}
