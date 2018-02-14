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

/* This file contains the implementation of the BLAS-1 function ccopy */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void ccopy_main (struct cublasCcopyParams parms);

/*
 * void
 * cublasCcopy (int n, const cuComplex *x, int incx, cuComplex *y, int incy)
 *
 * copies the single-complex vector x to the single-complex vector y. For 
 * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if 
 * incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar 
 * way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 * y      single-complex vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * y      contains-single complex vector x
 *
 * Reference: http://www.netlib.org/blas/ccopy.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasCcopy (int n, const cuComplex *x, int incx,
                                     cuComplex *y, int incy)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasCcopyParams params;
    cudaError_t cudaStat;
    int nbrCtas;
    int elemsPerCta;
    int threadsPerCta;

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }

    /* early out if nothing to do */
    if (n <= 0) {
        return;
    }

    memset (&params, 0, sizeof(params));
    params.n  = n;
    params.cx = x;
    params.incx = incx;
    params.cy = y;
    params.incy = incy;

    cublasVectorSplay (n, CUBLAS_CCOPY_THREAD_MIN, CUBLAS_CCOPY_THREAD_MAX,
                       CUBLAS_CCOPY_CTAS_MAX, &nbrCtas, &elemsPerCta, 
                       &threadsPerCta);
    
    cudaStat = cudaGetLastError(); /* clear error status */
    ccopy_main<<<nbrCtas,threadsPerCta>>>(params);
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

__global__ void ccopy_main (struct cublasCcopyParams parms) 
{
    int i, n, tid, totalThreads, ctaStart;
    const cuComplex *cx;
    cuComplex *cy;

    /* NOTE: host wrapper must ensure that parms.n > 0  */

    tid = threadIdx.x;
    n = parms.n;
    cx = parms.cx;
    cy = parms.cy;
    totalThreads = gridDim.x * blockDim.x;
    ctaStart = blockDim.x * blockIdx.x;
    
    if (parms.incy == 0) {
        if ((blockIdx.x == 0) && (tid == 0)) {
            int ix = (parms.incx > 0) ? ((parms.n - 1) * parms.incx) : 0;
            cy[0] = cx[ix];
        }
    } else if ((parms.incx == parms.incy) && (parms.incx > 0)) {
        /* equal, positive, increments */
        if (parms.incx == 1) {
            /* both increments equal to 1 */
            for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
                cy[i] = cx[i];
            }
        } else {
            /* equal, positive, non-unit increments. */
            for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
                cy[i*parms.incx] = cx[i*parms.incx];
            }
        }
    } else {
        /* unequal or nonpositive increments */
        int ix = ((parms.incx < 0) ? ((1 - n) * parms.incx) : 0);
        int iy = ((parms.incy < 0) ? ((1 - n) * parms.incy) : 0);
        for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
            cy[iy+i*parms.incy] = cx[ix+i*parms.incx];
        }
    }
}

