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

/* This file contains the implementation of the BLAS-2 function stbsv */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void stbsv_main_up_tr (struct cublasStbsvParams parms);
__global__ void stbsv_main_lo_tr (struct cublasStbsvParams parms);
__global__ void stbsv_main_up_nt (struct cublasStbsvParams parms);
__global__ void stbsv_main_lo_nt (struct cublasStbsvParams parms);

/*
 * void cublasStbsv (char uplo, char trans, char diag, int n, int k,
 *                   const float *A, int lda, float *X, int incx)
 *
 * solves one of the systems of equations op(A)*x = b, where op(A) is either 
 * op(A) = A or op(A) = transpose(A). b and x are n element vectors, and A is
 * an n x n unit or non-unit, upper or lower triangular band matrix with k + 1
 * diagonals. No test for singularity or near-singularity is included in this
 * function. Such tests must be performed before calling this function.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix is an upper or lower triangular band 
 *        matrix as follows: If uplo == 'U' or 'u', A is an upper triangular
 *        band matrix. If uplo == 'L' or 'l', A is a lower triangular band
 *        matrix.
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', 'C', or 'c', op(A) = transpose(A).
 * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
 *        assumed to be unit triangular; thas is, diagonal elements are not
 *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
 *        assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
 *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
 *        'l', k specifies the number of sub-diagonals. k must at least be 
 *        zero.
 * A      single precision array of dimension (lda, n). If uplo == 'U' or 'u',
 *        the leading (k + 1) x n part of the array A must contain the upper
 *        triangular band matrix, supplied column by column, with the leading
 *        diagonal of the matrix in row (k + 1) of the array, the first super-
 *        diagonal starting at position 2 in row k, and so on. The top left 
 *        k x k triangle of the array A is not referenced. If uplo == 'L' or 
 *        'l', the leading (k + 1) x n part of the array A must constain the
 *        lower triangular band matrix, supplied column by column, with the
 *        leading diagonal of the matrix in row 1 of the array, the first
 *        sub-diagonal starting at position 1 in row 2, and so on. The bottom
 *        right k x k triangle of the array is not referenced.
 * x      single precision array of length at least (1+(n-1)*abs(incx)).
 * incx   storage spacing between elements of x. It must not be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/stbsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n < 0, or n > 4070
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasStbsv (char uplo, char trans, char diag, int n,
                                     int k, const float *A, int lda, float *x,
                                     int incx)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasStbsvParams params;
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
    else if ((n < 0) || (n > CUBLAS_STBSV_MAX_DIM)) {
        info = 4;
    }
    else if (k < 0) {
        info = 5;
    }
    else if (lda < (k + 1)) {
        info = 7;
    }
    else if (incx == 0) {
        info = 9;
    }
    if (info) {
        cublasXerbla ("STBSV ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if (n == 0) {
        return;
    }

    memset (&params, 0, sizeof(params));
    params.k = k;
    params.n = n;
    params.A = A;
    params.x = x;
    params.lda = lda;
    params.incx = incx;
    params.trans = ((toupper(trans) == 'T') || (toupper(trans) == 'C'));
    params.unit = (toupper(diag) == 'U');
    params.up = (toupper(uplo) == 'U');

    cudaStat = cudaGetLastError(); /* clear error status */
    if (params.up) {
        if (params.trans) {
            stbsv_main_up_tr<<<CUBLAS_STBSV_CTAS,CUBLAS_STBSV_THREAD_COUNT>>>(params);
        } else {
            stbsv_main_up_nt<<<CUBLAS_STBSV_CTAS,CUBLAS_STBSV_THREAD_COUNT>>>(params);
        }
    } else {
         if (params.trans) {
            stbsv_main_lo_tr<<<CUBLAS_STBSV_CTAS,CUBLAS_STBSV_THREAD_COUNT>>>(params);
        } else {
            stbsv_main_lo_nt<<<CUBLAS_STBSV_CTAS,CUBLAS_STBSV_THREAD_COUNT>>>(params);
        }
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

/* column-major ordering */
#define IDXX(i)             (startx + ((i) * parms.incx))
#define XINC                (CUBLAS_STBSV_THREAD_COUNT)

__shared__ float XX[CUBLAS_STBSV_MAX_DIM];  /* cached portion of vector x */
__shared__ float temp;                      /* current diagonal element */


__global__ void stbsv_main_up_tr (struct cublasStbsvParams parms)
{
#undef  TRANS
#undef  UP
#define TRANS 1
#define UP    1
#include "stbsv.h"                
}

__global__ void stbsv_main_lo_tr (struct cublasStbsvParams parms)
{
#undef  TRANS
#undef  UP
#define TRANS 1
#define UP    0
#include "stbsv.h"                
}

__global__ void stbsv_main_up_nt (struct cublasStbsvParams parms)
{
#undef  TRANS
#undef  UP
#define TRANS 0
#define UP    1
#include "stbsv.h"                
}

__global__ void stbsv_main_lo_nt (struct cublasStbsvParams parms)
{
#undef  TRANS
#undef  UP
#define TRANS 0
#define UP    0
#include "stbsv.h"                
}


