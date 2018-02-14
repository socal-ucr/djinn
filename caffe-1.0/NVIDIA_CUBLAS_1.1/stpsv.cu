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

/* This file contains the implementation of the BLAS-2 function stpsv */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void stpsv_main_up_tr (struct cublasStpsvParams parms);
__global__ void stpsv_main_lo_tr (struct cublasStpsvParams parms);
__global__ void stpsv_main_up_nt (struct cublasStpsvParams parms);
__global__ void stpsv_main_lo_nt (struct cublasStpsvParams parms);

/*
 * void 
 * cublasStpsv (char uplo, char trans, char diag, int n, const float *AP, 
 *              float *X, int incx)
 *
 * solves one of the systems of equations op(A)*x = b, where op(A) is either 
 * op(A) = A or op(A) = transpose(A). b and x are n element vectors, and A is
 * an n x n unit or non-unit, upper or lower triangular matrix. No test for
 * singularity or near-singularity is included in this routine. Such tests 
 * must be performed before calling this routine.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix is an upper or lower triangular matrix
 *        as follows: If uplo == 'U' or 'u', A is an upper triangluar matrix.
 *        If uplo == 'L' or 'l', A is a lower triangular matrix.
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', 'C', or 'c', op(A) = transpose(A).
 * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
 *        assumed to be unit triangular; thas is, diagonal elements are not
 *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
 *        assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * AP     single precision array with at least ((n*(n+1))/2) elements. If uplo
 *        == 'U' or 'u', the array AP contains the upper triangular matrix A,
 *        packed sequentially, column by column; that is, if i <= j, then 
 *        A[i,j] is stored is AP[i+(j*(j+1)/2)]. If uplo == 'L' or 'L', the 
 *        array AP contains the lower triangular matrix A, packed sequentially,
 *        column by column; that is, if i >= j, then A[i,j] is stored in 
 *        AP[i+((2*n-j+1)*j)/2]. When diag = 'U' or 'u', the diagonal elements
 *        of A are not referenced and are assumed to be unity.
 * x      single precision array of length at least (1+(n-1)*abs(incx)).
 * incx   storage spacing between elements of x. It must not be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/stpsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0 or n > 4070
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasStpsv (char uplo, char trans, char diag, int n,
                                     const float *AP, float *x, int incx)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasStpsvParams params;
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
    else if ((n < 0) || (n > CUBLAS_STPSV_MAX_DIM)) {
        info = 4;
    }
    else if (incx == 0) {
        info = 7;
    }
    if (info) {
        cublasXerbla ("STPSV ", info);
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
    if (params.up) {
        if (params.trans) {
            stpsv_main_up_tr<<<CUBLAS_STPSV_CTAS,CUBLAS_STPSV_THREAD_COUNT>>>(params);
        } else {
            stpsv_main_up_nt<<<CUBLAS_STPSV_CTAS,CUBLAS_STPSV_THREAD_COUNT>>>(params);
        }
    } else {
         if (params.trans) {
            stpsv_main_lo_tr<<<CUBLAS_STPSV_CTAS,CUBLAS_STPSV_THREAD_COUNT>>>(params);
        } else {
            stpsv_main_lo_nt<<<CUBLAS_STPSV_CTAS,CUBLAS_STPSV_THREAD_COUNT>>>(params);
        }
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

/* column-major ordering */
#define IDXX(i)             (startx + ((i) * parms.incx))
#define XINC                (CUBLAS_STPSV_THREAD_COUNT)

__shared__ float XX[CUBLAS_STPSV_MAX_DIM];  /* cached portion of vector x */
__shared__ float temp;                      /* current diagonal element */


__global__ void stpsv_main_up_tr (struct cublasStpsvParams parms)
{
#undef  TRANS
#undef  UP
#define TRANS 1
#define UP    1
#include "stpsv.h"                
}

__global__ void stpsv_main_lo_tr (struct cublasStpsvParams parms)
{
#undef  TRANS
#undef  UP
#define TRANS 1
#define UP    0
#include "stpsv.h"                
}

__global__ void stpsv_main_up_nt (struct cublasStpsvParams parms)
{
#undef  TRANS
#undef  UP
#define TRANS 0
#define UP    1
#include "stpsv.h"                
}

__global__ void stpsv_main_lo_nt (struct cublasStpsvParams parms)
{
#undef  TRANS
#undef  UP
#define TRANS 0
#define UP    0
#include "stpsv.h"                
}


