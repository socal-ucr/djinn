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

/* This file contains the implementation of the BLAS-1 function csrot */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void csrot_main (struct cublasCsrotParams parms);

/*
 * void 
 * csrot (int n, cuComplex *x, int incx, cuCumplex *y, int incy, float c, 
 *        float s)
 *
 * multiplies a 2x2 matrix ( c s) with the 2xn matrix ( transpose(x) )
 *                         (-s c)                     ( transpose(y) )
 *
 * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if 
 * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and 
 * incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single-precision complex vector with n elements
 * incx   storage spacing between elements of x
 * y      single-precision complex vector with n elements
 * incy   storage spacing between elements of y
 * c      cosine component of rotation matrix
 * s      sine component of rotation matrix
 *
 * Output
 * ------
 * x      rotated vector x (unchanged if n <= 0)
 * y      rotated vector y (unchanged if n <= 0)
 *
 * Reference  http://www.netlib.org/blas/csrot.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasCsrot (int n, cuComplex *x, int incx, 
                                     cuComplex *y, int incy, float c, float s)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasCsrotParams params;
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
    params.sc = c;
    params.ss = s;

    cublasVectorSplay (n, CUBLAS_CSROT_THREAD_MIN, CUBLAS_CSROT_THREAD_MAX,
                       CUBLAS_CSROT_CTAS_MAX, &nbrCtas, &elemsPerCta, 
                       &threadsPerCta);

    cudaStat = cudaGetLastError(); /* clear error status */
    csrot_main<<<nbrCtas,threadsPerCta>>>(params);
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

__global__ void csrot_main (struct cublasCsrotParams parms) 
{
    int i, n, tid, totalThreads, ctaStart;
    cuComplex w, z;
    cuComplex *cx;
    cuComplex *cy;

    /* NOTE: wrapper must ensure that parms.n > 0  */

    tid = threadIdx.x;
    n = parms.n;
    cx = parms.cx; 
    cy = parms.cy;
    totalThreads = gridDim.x*blockDim.x;
    ctaStart = blockDim.x*blockIdx.x;
   
    if ((parms.incx == 0) || (parms.incy == 0)) {
        if ((blockIdx.x == 0) && (tid == 0)) {
            if ((parms.incx == 0) && (parms.incy == 0)) {
                cuComplex tw, tz;
                w = parms.cx[0];
                z = parms.cy[0];
                for (i = 0; i < parms.n; i++) {
                    tw.x = parms.sc * w.x + parms.ss * z.x;
                    tw.y = parms.sc * w.y + parms.ss * z.y;
                    tz.x = parms.sc * z.x - parms.ss * w.x;
                    tz.y = parms.sc * z.y - parms.ss * w.y;
                    w = tw;
                    z = tz;
                }
                cx[0] = w;
                cy[0] = z;
            } else if (parms.incx == 0) {
                int ky = (parms.incy < 0) ? ((1 - parms.n) * parms.incy) : 0;
                cuComplex temp = parms.cx[0];
                cuComplex tmp2;
                for (i = 0; i < parms.n; i++) {
                    w = temp;
                    z = cy[ky];
                    temp.x = parms.sc * w.x + parms.ss * z.x;
                    temp.y = parms.sc * w.y + parms.ss * z.y;
                    tmp2.x = parms.sc * z.x - parms.ss * w.x;
                    tmp2.y = parms.sc * z.y - parms.ss * w.y;
                    cy[ky] = tmp2;
                    ky += parms.incy;
                }
                cx[0] = temp;
            } else {
                int kx = (parms.incx < 0) ? ((1 - parms.n) * parms.incx) : 0;
                cuComplex temp = parms.cy[0];
                cuComplex tmp2;
                for (i = 0; i < parms.n; i++) {
                    w = parms.cx[kx];
                    z = temp;
                    tmp2.x = parms.sc * w.x + parms.ss * z.x;
                    tmp2.y = parms.sc * w.y + parms.ss * z.y;
                    temp.x = parms.sc * z.x - parms.ss * w.x;
                    temp.y = parms.sc * z.y - parms.ss * w.y;
                    parms.cx[kx] = tmp2;
                    kx += parms.incx;
                }
                parms.cy[0] = temp;
            }
        }
    } else if ((parms.incx == parms.incy) && (parms.incx > 0)) {
        /* equal, positive, increments */
        if (parms.incx == 1) {
            cuComplex temp, tmp2;
            /* both increments equal to 1 */
            for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
                w = cx[i];
                z = cy[i];
                temp.x = parms.sc * w.x + parms.ss * z.x;
                temp.y = parms.sc * w.y + parms.ss * z.y;
                tmp2.x = parms.sc * z.x - parms.ss * w.x;
                tmp2.y = parms.sc * z.y - parms.ss * w.y;
                cx[i] = temp;
                cy[i] = tmp2;
            }
        } else {
            /* equal, positive, non-unit increments. */
            cuComplex temp, tmp2;
            for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
                w = cx[i*parms.incx];
                z = cy[i*parms.incx];
                temp.x = parms.sc * w.x + parms.ss * z.x;
                temp.y = parms.sc * w.y + parms.ss * z.y;
                tmp2.x = parms.sc * z.x - parms.ss * w.x;
                tmp2.y = parms.sc * z.y - parms.ss * w.y;
                cx[i*parms.incx] = temp;
                cy[i*parms.incy] = tmp2;
            }
        }
    } else {
        /* unequal or nonpositive increments */
        cuComplex temp, tmp2;
        int ix = ((parms.incx < 0) ? ((1 - n) * parms.incx) : 0);
        int iy = ((parms.incy < 0) ? ((1 - n) * parms.incy) : 0);
        for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
            w = cx[ix+i*parms.incx];
            z = cy[iy+i*parms.incy];
            temp.x = parms.sc * w.x + parms.ss * z.x;
            temp.y = parms.sc * w.y + parms.ss * z.y;
            tmp2.x = parms.sc * z.x - parms.ss * w.x;
            tmp2.y = parms.sc * z.y - parms.ss * w.y;
            cx[ix+i*parms.incx] = temp;
            cy[iy+i*parms.incy] = tmp2;
        }
    }
}

