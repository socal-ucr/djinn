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

/* This file contains the implementation of the BLAS-2 function ssyr2 */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void ssyr2_up_main (struct cublasSsyr2Params parms);
__global__ void ssyr2_lo_main (struct cublasSsyr2Params parms);

/*
 * void cublasSsyr2 (char uplo, int n, float alpha, const float *x, int incx, 
 *                   const float *y, int incy, float *A, int lda)
 *
 * performs the symmetric rank 2 operation
 *
 *    A = alpha*x*transpose(y) + alpha*y*transpose(x) + A,
 *
 * where alpha is a single precision scalar, x and y are n element single 
 * precision vector and A is an n by n symmetric matrix consisting of single 
 * precision elements.
 * 
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array A. If uplo == 'U' or 'u', then only the 
 *        upper triangular part of A may be referenced and the lower triangular
 *        part of A is inferred. If uplo == 'L' or 'l', then only the lower 
 *        triangular part of A may be referenced and the upper triangular part
 *        of A is inferred.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  single precision scalar multiplier applied to x * transpose(y) + 
 *        y * transpose(x).
 * x      single precision array of length at least (1 + (n - 1) * abs (incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * y      single precision array of length at least (1 + (n - 1) * abs (incy)).
 * incy   storage spacing between elements of y. incy must not be zero.
 * A      single precision array of dimensions (lda, n). If uplo == 'U' or 'u',
 *        then A must contains the upper triangular part of a symmetric matrix,
 *        and the strictly lower triangular parts is not referenced. If uplo ==
 *        'L' or 'l', then A contains the lower triangular part of a symmetric 
 *        matrix, and the strictly upper triangular part is not referenced.
 * lda    leading dimension of A. It must be at least max(1, n).
 *
 * Output
 * ------
 * A      updated according to A = alpha*x*transpose(y)+alpha*y*transpose(x)+A
 *
 * Reference: http://www.netlib.org/blas/ssyr2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 * 
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSsyr2 (char uplo, int n, float alpha,
                                     const float *x, int incx, const float *y,
                                     int incy, float *A, int lda)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasSsyr2Params params;
    cudaError_t cudaStat;
    int info = 0;
    dim3 ctaDims(CUBLAS_SSYR2_GRIDW, CUBLAS_SSYR2_GRIDH);

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }
    info = 0;
    if ((toupper (uplo) != 'U') &&
        (toupper (uplo) != 'L')) {
        info = 1;
    }
    else if (n < 0) {
        info = 2;
    }
    else if (incx == 0) {
        info = 5;
    }
    else if (incy == 0) {
        info = 7;
    }
    else if (lda < imax (1, n)) {
        info = 9;
    }
    if (info) {
        cublasXerbla ("SSYR2 ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((n == 0) || (alpha == 0.0f)) {
        return;
    }

    memset (&params, 0, sizeof(params));
    params.up = toupper(uplo) == 'U';
    params.n = n;
    params.alpha = alpha;
    params.A = A;
    params.lda = lda;
    params.x = x;
    params.incx = incx;
    params.y = y;
    params.incy = incy;
    
    cudaStat = cudaGetLastError(); /* clear error status */
    if (params.up) {
        ssyr2_up_main<<<ctaDims,CUBLAS_SSYR2_THREAD_COUNT>>>(params);
    } else {
        ssyr2_lo_main<<<ctaDims,CUBLAS_SSYR2_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    params.x = y;
    params.incx = incy;
    params.y = x;
    params.incy = incx;

    cudaStat = cudaGetLastError(); /* clear error status */
    if (params.up) {
        ssyr2_up_main<<<ctaDims,CUBLAS_SSYR2_THREAD_COUNT>>>(params);
    } else {
        ssyr2_lo_main<<<ctaDims,CUBLAS_SSYR2_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

/* column-major ordering */
#undef IDXA
#undef IDXX
#undef IDXY
#define IDXA(row,col)       (parms.lda*(col)+(row))
#define IDXX(i)             (startx + ((i) * parms.incx))
#define IDXY(j)             (starty + ((j) * parms.incy))
#define BLK_LOG             (5)
#define BLK                 (1 << BLK_LOG)
#define ELEMS_PER_THREAD    ((BLK*BLK)/CUBLAS_SSYR2_THREAD_COUNT)
#define IIINC               (BLK)
#define JJINC               (IIINC)
#define IINC                (IIINC*CUBLAS_SSYR2_GRIDH)
#define JINC                (JJINC*CUBLAS_SSYR2_GRIDW)
#define A_NBR_COLS          (CUBLAS_SSYR2_THREAD_COUNT/IIINC)

#if (BLK & (BLK - 1))
#error tile dimension must be a power of two
#endif

#if (CUBLAS_SSYR2_THREAD_COUNT < BLK)
#error thread count must be greater than or equal to tile dimension
#endif

#if ((BLK*BLK)%CUBLAS_SSYR2_THREAD_COUNT)
#error number of tile elements must be integral multiple of thread count
#endif

#if (CUBLAS_SSYR2_THREAD_COUNT%IIINC)
#error thread count must be integral multiple of tile dimension
#endif

__shared__ float xi[IINC];
__shared__ float yj[JINC];

__global__ void ssyr2_up_main (struct cublasSsyr2Params parms) 
{
#undef LOWER
#define LOWER 0
#include "ssyr2.h"
}

__global__ void ssyr2_lo_main (struct cublasSsyr2Params parms) 
{
#undef LOWER
#define LOWER 1
#include "ssyr2.h"
}
