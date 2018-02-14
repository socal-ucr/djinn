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

/* This file contains the implementation of the BLAS-2 function sspr */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void sspr_up_main (struct cublasSsprParams parms);
__global__ void sspr_lo_main (struct cublasSsprParams parms);

/*
 * void 
 * cublasSspr (char uplo, int n, float alpha, const float *x, int incx, 
 *             float *AP)
 *
 * performs the symmetric rank 1 operation
 * 
 *    A = alpha * x * transpose(x) + A,
 * 
 * where alpha is a single precision scalar and x is an n element single 
 * precision vector. A is a symmetric n x n matrix consisting of single 
 * precision elements that is supplied in packed form.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array AP. If uplo == 'U' or 'u', then the upper 
 *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then 
 *        the lower triangular part of A is supplied in AP.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  single precision scalar multiplier applied to x * transpose(x).
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part 
 *        of the symmetric matrix A, packed sequentially, column by column; 
 *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If 
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part 
 *        of the symmetric matrix A, packed sequentially, column by column; 
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * transpose(x) + A
 *
 * Reference: http://www.netlib.org/blas/sspr.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSspr (char uplo, int n, float alpha,
                                    const float *x, int incx, float *AP)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasSsprParams params;
    cudaError_t cudaStat;
    int up;
    int info = 0;
    dim3 ctaDims(CUBLAS_SSPR_GRIDW, CUBLAS_SSPR_GRIDH);

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }

    up = toupper(uplo) == 'U';

    info = 0;
    if ((!up) && (toupper (uplo) != 'L')) {
        info = 1;
    }
    else if (n < 0) {
        info = 2;
    }
    else if (incx == 0) {
        info = 5;
    }
    if (info) {
        cublasXerbla ("SSPR  ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((n == 0) || (alpha == 0.0f)) {
        return;
    }

    memset (&params, 0, sizeof(params));
    params.up = up;
    params.n = n;
    params.alpha = alpha;
    params.AP = AP;
    params.x = x;
    params.incx = incx;
    
    cudaStat = cudaGetLastError(); /* clear error status */
    if (params.up) {
        sspr_up_main<<<ctaDims,CUBLAS_SSPR_THREAD_COUNT>>>(params);
    } else {
        sspr_lo_main<<<ctaDims,CUBLAS_SSPR_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

/* column-major ordering */
#undef IDXX
#define IDXX(i)             (startx + ((i) * parms.incx))

#define BLK_LOG             (5)
#define BLK                 (1 << BLK_LOG)
#define ELEMS_PER_THREAD    ((BLK*BLK)/CUBLAS_SSPR_THREAD_COUNT)
#define IIINC               (BLK)
#define JJINC               (BLK)
#define IINC                (IIINC*CUBLAS_SSPR_GRIDH)
#define JINC                (JJINC*CUBLAS_SSPR_GRIDW)
#define A_NBR_COLS          (CUBLAS_SSPR_THREAD_COUNT/IIINC)

#if (BLK & (BLK - 1))
#error tile dimension must be a power of two
#endif

#if (CUBLAS_SSPR_THREAD_COUNT < BLK)
#error thread count must be greater than or equal to tile dimension
#endif

#if ((BLK*BLK)%CUBLAS_SSPR_THREAD_COUNT)
#error number of tile elements must be integral multiple of thread count
#endif

#if (CUBLAS_SSPR_THREAD_COUNT%IIINC)
#error thread count must be integral multple of tile dimension
#endif

__shared__ float xi[IINC];
__shared__ float xj[IINC];

__global__ void sspr_up_main (struct cublasSsprParams parms) 
{
#undef LOWER
#define LOWER 0
#include "sspr.h"
}

__global__ void sspr_lo_main (struct cublasSsprParams parms)
{
#undef LOWER
#define LOWER 1
#include "sspr.h"
}
