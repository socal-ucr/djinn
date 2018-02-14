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

/* This file contains the implementation of the BLAS-1 function csscal */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void csscal_main (struct cublasCsscalParams parms);

/*
 * void
 * cublasCsscal (int n, float alpha, cuComplex *x, int incx)
 *
 * replaces single-complex vector x with single-complex alpha * x. For i 
 * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx], 
 * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  single precision scalar multiplier
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * x      single-complex result (unchanged if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/csscal.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 * 
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasCsscal (int n, float alpha, cuComplex *x,
                                      int incx)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasCsscalParams params;
    cudaError_t cudaStat;
    int nbrCtas;
    int elemsPerCta;
    int threadsPerCta;

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }
    /* early out if nothing to do */
    if ((n <= 0) || (incx <= 0)) {
        return;
    }

    memset (&params, 0, sizeof(params));
    params.n  = n;
    params.cx = x;
    params.sa = alpha;
    params.incx = incx;

    cublasVectorSplay (n, CUBLAS_CSSCAL_THREAD_MIN, CUBLAS_CSSCAL_THREAD_MAX,
                       CUBLAS_CSSCAL_CTAS_MAX, &nbrCtas, &elemsPerCta, 
                       &threadsPerCta);

    cudaStat = cudaGetLastError(); /* clear error status */
    csscal_main<<<nbrCtas,threadsPerCta>>>(params);
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

__global__ void csscal_main (struct cublasCsscalParams parms)
{
    int i, n, tid, totalThreads, ctaStart;
    cuComplex *cx;

    /* NOTE: wrapper must ensure that parms.n > 0 and parms.incx > 0 */

    tid = threadIdx.x;
    n = parms.n;
    cx = parms.cx;
    totalThreads = gridDim.x * blockDim.x;
    ctaStart = blockDim.x * blockIdx.x;

    if (parms.incx == 1) {
        /* increment equal to 1 */
        for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
            cx[i] = make_cuComplex (parms.sa * cuCrealf(cx[i]), 
                                    parms.sa * cuCimagf(cx[i]));
        }
    } else {
        /* increment not equal to 1 */
        int ix = ((parms.incx < 0) ? ((1 - n) * parms.incx) : 0);
        for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
            cx[ix+i*parms.incx] = make_cuComplex (parms.sa * 
                                                  cuCrealf(cx[ix+i*parms.incx]), 
                                                  parms.sa * 
                                                  cuCimagf(cx[ix+i*parms.incx]));
        }
    }
}
