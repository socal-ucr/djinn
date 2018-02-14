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

/* This file contains the implementation of the BLAS-2 function strmv */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void strmv_up_tr_main (struct cublasStrmvParams parms);
__global__ void strmv_up_nt_main (struct cublasStrmvParams parms);
__global__ void strmv_lo_tr_main (struct cublasStrmvParams parms);
__global__ void strmv_lo_nt_main (struct cublasStrmvParams parms);

/*
 * void 
 * cublasStrmv (char uplo, char trans, char diag, int n, const float *A, 
 *              int lda, float *x, int incx);
 *
 * performs one of the matrix-vector operations x = op(A) * x, where op(A) = 
 = A, or op(A) = transpose(A). x is an n-element single precision vector, and 
 * A is an n x n, unit or non-unit, upper or lower, triangular matrix composed 
 * of single precision elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular 
 *        matrix. If uplo = 'U' or 'u', then A is an upper triangular matrix. 
 *        If uplo = 'L' or 'l', then A is a lower triangular matrix.
 * trans  specifies op(A). If transa = 'N' or 'n', op(A) = A. If trans = 'T', 
 *        't', 'C', or 'c', op(A) = transpose(A)
 * diag   specifies whether or not matrix A is unit triangular. If diag = 'U' 
 *        or 'u', A is assumed to be unit triangular. If diag = 'N' or 'n', A 
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be 
 *        at least zero.
 * A      single precision array of dimension (lda, n). If uplo = 'U' or 'u', 
 *        the leading n x n upper triangular part of the array A must contain 
 *        the upper triangular matrix and the strictly lower triangular part 
 *        of A is not referenced. If uplo = 'L' or 'l', the leading n x n lower
 *        triangular part of the array A must contain the lower triangular 
 *        matrix and the strictly upper triangular part of A is not referenced.
 *        When diag = 'U' or 'u', the diagonal elements of A are not referenced
 *        either, but are are assumed to be unity.
 * lda    is the leading dimension of A. It must be at least max (1, n).
 * x      single precision array of length at least (1 + (n - 1) * abs(incx) ).
 *        On entry, x contains the source vector. On exit, x is overwritten 
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be 
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x,
 *
 * Reference: http://www.netlib.org/blas/strmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasStrmv (char uplo, char trans, char diag, int n, 
                                     const float *A, int lda, float *x,
                                     int incx)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasStrmvParams params;
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
    else if ((n < 0) && (n > CUBLAS_STRMV_MAX_DIM)) {
        info = 4;
    }
    else if (lda < imax (1, n)) {
        info = 6;
    }
    else if (incx == 0) {
        info = 8;
    }
    if (info) {
        cublasXerbla ("STRMV ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if (n == 0) {
        return;
    }

    memset (&params, 0, sizeof(params));
    params.n = n;
    params.A = A;
    params.lda = lda;
    params.x = x;
    params.incx = incx;
    params.trans = ((toupper(trans) == 'T') || (toupper(trans) == 'C'));
    params.unit = (toupper(diag) == 'U');
    params.up = (toupper(uplo) == 'U');

    cudaStat = cudaGetLastError(); /* clear error status */
    if (params.up) {
        if (params.trans) {
            strmv_up_tr_main<<<CUBLAS_STRMV_CTAS,CUBLAS_STRMV_THREAD_COUNT>>>(params);
        } else {
            strmv_up_nt_main<<<CUBLAS_STRMV_CTAS,CUBLAS_STRMV_THREAD_COUNT>>>(params);
        }
    } else {
        if (params.trans) {
            strmv_lo_tr_main<<<CUBLAS_STRMV_CTAS,CUBLAS_STRMV_THREAD_COUNT>>>(params);
        } else {
            strmv_lo_nt_main<<<CUBLAS_STRMV_CTAS,CUBLAS_STRMV_THREAD_COUNT>>>(params);
        }
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

/* column-major ordering */
#define IDXA(row,col)       (parms.lda*(col)+(row))
#define IDXX(i)             (startx + ((i) * parms.incx))
#define XINC                (CUBLAS_STRMV_THREAD_COUNT)
#define COL_OFS             (IDXA(0,1)-IDXA(0,0))
#define ROW_OFS             (IDXA(1,0)-IDXA(0,0))

__shared__ float xx[CUBLAS_STRMV_MAX_DIM];  /* cached portion of vector x */

__global__ void strmv_up_tr_main (struct cublasStrmvParams parms) 
{
#undef UP
#undef TRANS
#define TRANS 1
#define UP    1
#include "strmv.h"
}

__global__ void strmv_up_nt_main (struct cublasStrmvParams parms) 
{
#undef UP
#undef TRANS
#define TRANS 0
#define UP    1
#include "strmv.h"
}

__global__ void strmv_lo_tr_main (struct cublasStrmvParams parms) 
{
#undef UP
#undef TRANS
#define TRANS 1
#define UP    0
#include "strmv.h"
}

__global__ void strmv_lo_nt_main (struct cublasStrmvParams parms) 
{
#undef UP
#undef TRANS
#define TRANS 0
#define UP    0
#include "strmv.h"
}
