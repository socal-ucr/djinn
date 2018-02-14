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

/* This file contains the implementation of the BLAS-2 function stpmv */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void stpmv_main_up_tr (struct cublasStpmvParams parms);
__global__ void stpmv_main_lo_tr (struct cublasStpmvParams parms);
__global__ void stpmv_main_up_nt (struct cublasStpmvParams parms);
__global__ void stpmv_main_lo_nt (struct cublasStpmvParams parms);

/*
 * void 
 * cublasStpmv (char uplo, char trans, char diag, int n, const float *AP, 
 *              float *x, int incx);
 *
 * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
 * or op(A) = transpose(A). x is an n element single precision vector, and A 
 * is an n x n, unit or non-unit, upper or lower triangular matrix composed 
 * of single precision elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo == 'U' or 'u', then A is an upper triangular matrix.
 *        If uplo == 'L' or 'l', then A is a lower triangular matrix.
 * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', 'C', or 'c', op(A) = transpose(A)
 * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
 *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A 
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be 
 *        at least zero. In the current implementation n must not exceed 4070.
 * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part 
 *        of the symmetric matrix A, packed sequentially, column by column; 
 *        that is, if i <= j, then A[i,j] is stored in AP[i+(j*(j+1)/2)]. If 
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part 
 *        of the symmetric matrix A, packed sequentially, column by column; 
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the source vector. On exit, x is overwritten 
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be 
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x,
 *
 * Reference: http://www.netlib.org/blas/stpmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n < 0, or n > 4070
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasStpmv (char uplo, char trans, char diag, int n, 
                                     const float *AP, float *x, int incx)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasStpmvParams params;
    cudaError_t cudaStat;
    int info = 0;

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }

    /* check other inputs */
    if ((toupper (uplo) != 'U') &&
        (toupper (uplo) != 'L')) {
        info = 1;
    }
    else if ((toupper (trans) != 'N') && 
             (toupper (trans) != 'T') && 
             (toupper (trans) != 'C')) {
        info = 2;
    } 
    else if ((toupper (diag) != 'U') &&
             (toupper (diag) != 'N')) {
        info = 3;
    }
    else if ((n < 0) && (n > CUBLAS_STPMV_MAX_DIM)) {
        info = 4;
    }
    else if (incx == 0) {
        info = 7;
    }
    if (info) {
        cublasXerbla ("STPMV ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if (n == 0) {
        return;
    }

    memset (&params, 0, sizeof(params));
    params.n = n;
    params.AP = AP;
    params.x = x;
    params.incx = incx;
    params.trans = ((toupper(trans) == 'T') || (toupper(trans) == 'C'));
    params.unit = (toupper(diag) == 'U');
    params.up = (toupper(uplo) == 'U');

    cudaStat = cudaGetLastError(); /* clear error status */
    if (params.trans) {
        if (params.up) {
            stpmv_main_up_tr<<<CUBLAS_STPMV_CTAS,CUBLAS_STPMV_THREAD_COUNT>>>(params);
        } else {
            stpmv_main_lo_tr<<<CUBLAS_STPMV_CTAS,CUBLAS_STPMV_THREAD_COUNT>>>(params);
        }
    } else {
        if (params.up) {
            stpmv_main_up_nt<<<CUBLAS_STPMV_CTAS,CUBLAS_STPMV_THREAD_COUNT>>>(params);
        } else {
            stpmv_main_lo_nt<<<CUBLAS_STPMV_CTAS,CUBLAS_STPMV_THREAD_COUNT>>>(params);
        }
    }
    cudaStat = cudaGetLastError(); /* check for launch error */
    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

/* column-major ordering */
#define IDXX(i)             (startx + ((i) * parms.incx))
#define XINC                (CUBLAS_STPMV_THREAD_COUNT)

__shared__ float xx[CUBLAS_STPMV_MAX_DIM];  /* cached portion of vector x */

__global__ void stpmv_main_up_tr (struct cublasStpmvParams parms) 
{
#undef  UP
#undef  TRANS
#define TRANS 1
#define UP    1
#include "stpmv.h"
}

__global__ void stpmv_main_lo_tr (struct cublasStpmvParams parms) 
{
#undef  UP
#undef  TRANS
#define TRANS 1
#define UP    0
#include "stpmv.h"
}

__global__ void stpmv_main_up_nt (struct cublasStpmvParams parms) 
{
#undef  UP
#undef  TRANS
#define TRANS 0
#define UP    1
#include "stpmv.h"
}

__global__ void stpmv_main_lo_nt (struct cublasStpmvParams parms) 
{
#undef  UP
#undef  TRANS
#define TRANS 0
#define UP    0
#include "stpmv.h"
}
