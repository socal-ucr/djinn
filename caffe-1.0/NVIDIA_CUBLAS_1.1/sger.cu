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

/* This file contains the implementation of the BLAS-2 function sger */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

#define TILE_DIM_LOG        (5)
#define TILE_DIM            (1 << TILE_DIM_LOG)
#define TILE_SIZE           ((TILE_DIM) * (TILE_DIM))
#define ELEMS_PER_THREAD    ((TILE_SIZE)/(CUBLAS_SGER_THREAD_COUNT))
#define SUP_TILE_DIM        (TILE_DIM*CUBLAS_SGER_GRIDH)
#define A_NBR_COLS          (CUBLAS_SGER_THREAD_COUNT/TILE_DIM)

#if (TILE_DIM & (TILE_DIM - 1))
#error tile dimension must be a power of two
#endif

#if (CUBLAS_SGER_THREAD_COUNT < TILE_DIM)
#error thread count must be greater than or equal to tile dimension
#endif

#if ((TILE_SIZE)%CUBLAS_SGER_THREAD_COUNT)
#error number of tile elements must be integral multiple of thread count
#endif

#if (CUBLAS_SGER_THREAD_COUNT%TILE_DIM)
#error thread count must be integral multple of tile dimension
#endif

__global__ void sger_main_hw (struct cublasSgerParams parms);
__global__ void sger_main_fast_hw (struct cublasSgerParams parms);
__global__ void sger_main_sw (struct cublasSgerParams parms);

/*
 * cublasSger (int m, int n, float alpha, const float *x, int incx, 
 *             const float *y, int incy, float *A, int lda)
 *
 * performs the symmetric rank 1 operation
 *
 *    A = alpha * x * transpose(y) + A,
 *
 * where alpha is a single precision scalar, x is an m element single 
 * precision vector, y is an n element single precision vector, and A 
 * is an m by n matrix consisting of single precision elements. Matrix A
 * is stored in column major format, and lda is the leading dimension of
 * the two-dimensional array used to store A.
 * 
 * Input
 * -----
 * m      specifies the number of rows of the matrix A. It must be at least 
 *        zero.
 * n      specifies the number of columns of the matrix A. It must be at 
 *        least zero.
 * alpha  single precision scalar multiplier applied to x * transpose(y)
 * x      single precision array of length at least (1 + (m - 1) * abs(incx))
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 * y      single precision array of length at least (1 + (n - 1) * abs(incy))
 * incy   specifies the storage spacing between elements of y. incy must not 
 *        be zero.
 * A      single precision array of dimensions (lda, n).
 * lda    leading dimension of two-dimensional array used to store matrix A
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * transpose(y) + A
 *
 * Reference: http://www.netlib.org/blas/sger.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 * 
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSger (int m, int n, float alpha, const float *x,
                                    int incx, const float *y, int incy,
                                    float *A, int lda)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasSgerParams params;
    cudaError_t cudaStat;
    int info = 0;
    int usePureHwStepper;
    int useFastImul;
    dim3 ctaDimsHw (((m+TILE_DIM-1)/TILE_DIM), ((n+TILE_DIM-1)/TILE_DIM));
    dim3 ctaDimsSw (CUBLAS_SGER_GRIDW, CUBLAS_SGER_GRIDH);

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }

    info = 0;
    if (m < 0) {
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
    else if (lda < imax (1, m)) {
        info = 9;
    }
    if (info) {
        cublasXerbla ("SGER  ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((m == 0)|| (n == 0) || (alpha == 0.0f)) {
        return;
    }

    memset (&params, 0, sizeof(params));
    params.m = m;
    params.n = n;
    params.alpha = alpha;
    params.A = A;
    params.lda = lda;
    params.x = x;
    params.incx = incx;
    params.y = y;
    params.incy = incy;

    /* choose version using 24-bit multiplies if all dimensions are less than
     * 2001, so we can guarantee that no multiplication result exceeds (2000 *
     * 2000 * 4) < 2^24. Increments must be positive since we use unsigned
     * multiplication
     */
    useFastImul = ((params.lda  <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                   (params.m    <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                   (params.n    <= CUBLAS_FASTIMUL_F_MAX_DIM) &&
                   (params.incx <= CUBLAS_FASTIMUL_F_MAX_DIM) &&
                   (params.incy <= CUBLAS_FASTIMUL_F_MAX_DIM) &&
                   (params.incx > 0) && (params.incy > 0));

    /* choose HW-only stepping if dimensions of result matrix do not exceed the
     * maximum CTA grid dimensions.
     */
    usePureHwStepper = ((m < (CUBLAS_CTA_MAX_DIM * TILE_DIM)) &&
                        (n < (CUBLAS_CTA_MAX_DIM * TILE_DIM)));
    
    cudaStat = cudaGetLastError(); /* clear error status */
    if (usePureHwStepper) {
        if (useFastImul) {
            sger_main_fast_hw<<<ctaDimsHw,CUBLAS_SGER_THREAD_COUNT>>>(params);
        } else {
            sger_main_hw<<<ctaDimsHw,CUBLAS_SGER_THREAD_COUNT>>>(params);
        }
    } else {
        sger_main_sw<<<ctaDimsSw,CUBLAS_SGER_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

__shared__ float xi[TILE_DIM];
__shared__ float yj[TILE_DIM];

__global__ void sger_main_hw (struct cublasSgerParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FAST_IMUL
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#include "sger.h"
}

__global__ void sger_main_fast_hw (struct cublasSgerParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FAST_IMUL
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#include "sger.h"
}

__global__ void sger_main_sw (struct cublasSgerParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FAST_IMUL
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#include "sger.h"
}
