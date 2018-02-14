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

/* This file contains the implementation of the BLAS-2 function ssyr */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

#if (CUBLAS_SSYR_GRIDW!=CUBLAS_SSYR_GRIDH)
#error super tile is not square!
#endif

#define TILE_DIM_LOG        (5)
#define TILE_DIM            (1 << (TILE_DIM_LOG))
#define TILE_SIZE           (TILE_DIM*TILE_DIM)
#define SUP_TILE_DIM        (TILE_DIM*CUBLAS_SSYR_GRIDW)

#if ((TILE_SIZE%CUBLAS_SSYR_THREAD_COUNT)!=0)
#error TILE_SIZE and CUBLAS_SSYR_THREAD_COUNT do not divide evenly!
#endif
#if ((CUBLAS_SSYR_THREAD_COUNT%TILE_DIM)!=0)
#error CUBLAS_SSYR_THREAD_COUNT and TILE_DIM do not divide evenly!
#endif

#define COL_INCR            (CUBLAS_SSYR_THREAD_COUNT/TILE_DIM)
#define ELEMS_PER_THREAD    (TILE_SIZE/CUBLAS_SSYR_THREAD_COUNT)

__global__ void ssyr_main_up_sw (struct cublasSsyrParams parms);
__global__ void ssyr_main_lo_sw (struct cublasSsyrParams parms);
__global__ void ssyr_main_up_hw (struct cublasSsyrParams parms);
__global__ void ssyr_main_lo_hw (struct cublasSsyrParams parms);

/*
 * void 
 * cublasSsyr (char uplo, int n, float alpha, const float *x, int incx, 
 *             float *A, int lda)
 *
 * performs the symmetric rank 1 operation
 *
 *    A = alpha * x * transpose(x) + A,
 *
 * where alpha is a single precision scalar, x is an n element single 
 * precision vector and A is an n x n symmetric matrix consisting of 
 * single precision elements. Matrix A is stored in column major format,
 * and lda is the leading dimension of the two-dimensional array 
 * containing A.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or 
 *        the lower triangular part of array A. If uplo = 'U' or 'u',
 *        then only the upper triangular part of A may be referenced.
 *        If uplo = 'L' or 'l', then only the lower triangular part of
 *        A may be referenced.
 * n      specifies the number of rows and columns of the matrix A. It
 *        must be at least 0.
 * alpha  single precision scalar multiplier applied to x * transpose(x)
 * x      single precision array of length at least (1 + (n - 1) * abs(incx))
 * incx   specifies the storage spacing between elements of x. incx must 
 *        not be zero.
 * A      single precision array of dimensions (lda, n). If uplo = 'U' or 
 *        'u', then A must contain the upper triangular part of a symmetric 
 *        matrix, and the strictly lower triangular part is not referenced. 
 *        If uplo = 'L' or 'l', then A contains the lower triangular part 
 *        of a symmetric matrix, and the strictly upper triangular part is 
 *        not referenced.
 * lda    leading dimension of the two-dimensional array containing A. lda
 *        must be at least max(1, n).
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * transpose(x) + A
 *
 * Reference: http://www.netlib.org/blas/ssyr.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSsyr (char uplo, int n, float alpha,
                                    const float *x, int incx, float *A, 
                                    int lda)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasSsyrParams params;
    cudaError_t cudaStat;
    int up;
    int info = 0;
    int usePureHwStepper = 0;
    dim3 ctaDimsHw (((n+TILE_DIM-1)/TILE_DIM), ((n+TILE_DIM-1)/TILE_DIM));
    dim3 ctaDimsSw (CUBLAS_SSYR_GRIDW, CUBLAS_SSYR_GRIDH);

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
    else if (lda < imax (1, n)) {
        info = 7;
    }
    if (info) {
        cublasXerbla ("SSYR  ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((n == 0) || (alpha == 0.0f)) {
        return;
    }

    /* choose HW-only stepping if dimensions of result matrix do not exceed the
     * maximum CTA grid dimensions.
     */
    usePureHwStepper = n < (CUBLAS_CTA_MAX_DIM * TILE_DIM);

    /* HW grid never seems to be a win on G80, maybe because the supertile has 
       better spatial locality ?
    */
    usePureHwStepper = 0; 

    memset (&params, 0, sizeof(params));
    up = toupper(uplo) == 'U';

    params.up = up;
    params.n = n;
    params.alpha = alpha;
    params.A = A;
    params.lda = lda;
    params.x = x;
    params.incx = incx;
    
    cudaStat = cudaGetLastError(); /* clear error status */
    if (usePureHwStepper) {
        if (params.up) {
            ssyr_main_up_hw<<<ctaDimsHw,CUBLAS_SSYR_THREAD_COUNT>>>(params);
        } else {
            ssyr_main_lo_hw<<<ctaDimsHw,CUBLAS_SSYR_THREAD_COUNT>>>(params);
        }
    } else {
        if (params.up) {
            ssyr_main_up_sw<<<ctaDimsSw,CUBLAS_SSYR_THREAD_COUNT>>>(params);
        } else {
            ssyr_main_lo_sw<<<ctaDimsSw,CUBLAS_SSYR_THREAD_COUNT>>>(params);
        }
    }       
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

/* column-major ordering */
#undef IDXA
#undef IDXX
#define IDXA(row,col) (parms.lda*(col)+(row))
#define IDXX(i)       (startx+((i)*parms.incx))

__shared__ float xi[TILE_DIM];
__shared__ float xj[TILE_DIM];

__global__ void ssyr_main_up_hw (struct cublasSsyrParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  LOWER
#define LOWER              0
#define USE_MIXED_STEPPER  0
#include "ssyr.h"
}

__global__ void ssyr_main_lo_hw (struct cublasSsyrParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  LOWER
#define LOWER              1
#define USE_MIXED_STEPPER  0
#include "ssyr.h"
}

__global__ void ssyr_main_up_sw (struct cublasSsyrParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  LOWER
#define LOWER              0
#define USE_MIXED_STEPPER  1
#include "ssyr.h"
}

__global__ void ssyr_main_lo_sw (struct cublasSsyrParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  LOWER
#define LOWER              1
#define USE_MIXED_STEPPER  1
#include "ssyr.h"
}
