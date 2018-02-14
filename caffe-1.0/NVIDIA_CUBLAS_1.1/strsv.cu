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

/* This file contains the implementation of the BLAS-2 function strsv */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void strsv_main_tr_up (struct cublasStrsvParams parms);
__global__ void strsv_main_tr_lo (struct cublasStrsvParams parms);
__global__ void strsv_main_nt_up (struct cublasStrsvParams parms);
__global__ void strsv_main_nt_lo (struct cublasStrsvParams parms);

/*
 * void 
 * cublasStrsv (char uplo, char trans, char diag, int n, const float *A, 
 *              int lda, float *x, int incx)
 *
 * solves a system of equations op(A) * x = b, where op(A) is either A or 
 * transpose(A). b and x are single precision vectors consisting of n
 * elements, and A is an n x n matrix composed of a unit or non-unit, upper
 * or lower triangular matrix. Matrix A is stored in column major format,
 * and lda is the leading dimension of the two-diemnsional array containing
 * A.
 *
 * No test for singularity or near-singularity is included in this function. 
 * Such tests must be performed before calling this function.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the 
 *        lower triangular part of array A. If uplo = 'U' or 'u', then only 
 *        the upper triangular part of A may be referenced. If uplo = 'L' or 
 *        'l', then only the lower triangular part of A may be referenced.
 * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If transa = 't',
 *        'T', 'c', or 'C', op(A) = transpose(A)
 * diag   specifies whether or not A is a unit triangular matrix like so:
 *        if diag = 'U' or 'u', A is assumed to be unit triangular. If 
 *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. It
 *        must be at least 0. In the current implementation n must be <=
 *        4070.
 * A      is a single precision array of dimensions (lda, n). If uplo = 'U' 
 *        or 'u', then A must contains the upper triangular part of a symmetric
 *        matrix, and the strictly lower triangular parts is not referenced. 
 *        If uplo = 'L' or 'l', then A contains the lower triangular part of 
 *        a symmetric matrix, and the strictly upper triangular part is not 
 *        referenced. 
 * lda    is the leading dimension of the two-dimensional array containing A.
 *        lda must be at least max(1, n).
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the n element right-hand side vector b. On exit,
 *        it is overwritten with the solution vector x.
 * incx   specifies the storage spacing between elements of x. incx must not 
 *        be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/strsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0 or n > 4070
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasStrsv (char uplo, char trans, char diag, int n, 
                                     const float *A, int lda, float *x, 
                                     int incx)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasStrsvParams params;
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
    /* FIXME: There should be no upper limit on n */
    else if ((n < 0) || (n > CUBLAS_STRSV_MAX_DIM)) {
        info = 4;
    }
    else if (lda < imax (1, n)) {
        info = 6;
    }
    else if (incx == 0) {
        info = 8;
    }
    if (info) {
        cublasXerbla ("STRSV ", info);
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
    if (params.trans) {
        if (params.up) {
            strsv_main_tr_up<<<CUBLAS_STRSV_CTAS,CUBLAS_STRSV_THREAD_COUNT>>>(params);
        } else {
            strsv_main_tr_lo<<<CUBLAS_STRSV_CTAS,CUBLAS_STRSV_THREAD_COUNT>>>(params);
        }
    } else {
        if (params.up) {
            strsv_main_nt_up<<<CUBLAS_STRSV_CTAS,CUBLAS_STRSV_THREAD_COUNT>>>(params);
        } else {
            strsv_main_nt_lo<<<CUBLAS_STRSV_CTAS,CUBLAS_STRSV_THREAD_COUNT>>>(params);
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
#define XINC                (CUBLAS_STRSV_THREAD_COUNT)

__shared__ float XX[CUBLAS_STRSV_MAX_DIM];  /* cached portion of vector x */
__shared__ float temp;                      /* current diagonal element */

__global__ void strsv_main_tr_up (struct cublasStrsvParams parms) 
{
#undef  TRANS
#undef  UP
#define TRANS 1
#define UP    1
#include "strsv.h"
}
__global__ void strsv_main_tr_lo (struct cublasStrsvParams parms)
{
#undef  TRANS
#undef  UP
#define TRANS 1
#define UP    0
#include "strsv.h"
}

__global__ void strsv_main_nt_up (struct cublasStrsvParams parms)
{
#undef  TRANS
#undef  UP
#define TRANS 0
#define UP    1
#include "strsv.h"
}

__global__ void strsv_main_nt_lo (struct cublasStrsvParams parms)
{
#undef  TRANS
#undef  UP
#define TRANS 0
#define UP    0
#include "strsv.h"
}
