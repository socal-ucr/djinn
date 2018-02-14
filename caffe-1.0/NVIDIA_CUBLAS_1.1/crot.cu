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

/* This file contains the implementation of the BLAS-1 function crot */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void crot_main (struct cublasCrotParams parms);

/*
 * cublasCrot (int n, cuComplex *x, int incx, cuComplex *y, int incy, float sc,
 *             cuComplex cs)
 *
 * multiplies a 2x2 matrix ( sc       cs) with the 2xn matrix ( transpose(x) )
 *                         (-conj(cs) sc)                     ( transpose(y) )
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
 * sc     single-precision cosine component of rotation matrix
 * cs     single-precision complex sine component of rotation matrix
 *
 * Output
 * ------
 * x      rotated single-precision complex vector x (unchanged if n <= 0)
 * y      rotated single-precision complex vector y (unchanged if n <= 0)
 *
 * Reference: http://netlib.org/lapack/explore-html/crot.f.html
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasCrot (int n, cuComplex *x, int incx, 
                                    cuComplex *y, int incy, float sc, 
                                    cuComplex cs)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasCrotParams params;
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
    params.sc = sc;
    params.cs = cs;

    cublasVectorSplay (n, CUBLAS_CROT_THREAD_MIN, CUBLAS_CROT_THREAD_MAX,
                       CUBLAS_CROT_CTAS_MAX, &nbrCtas, &elemsPerCta, 
                       &threadsPerCta);

    cudaStat = cudaGetLastError(); /* clear error status */
    crot_main<<<nbrCtas,threadsPerCta>>>(params);
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

__global__ void crot_main (struct cublasCrotParams parms) 
{
    int i, n, tid, totalThreads, ctaStart;
    cuComplex w, z, conjugCs;
    cuComplex *cx;
    cuComplex *cy;
    cuComplex cc;

    /* NOTE: wrapper must ensure that parms.n > 0  */

    tid = threadIdx.x;
    n = parms.n;
    cx = parms.cx; 
    cy = parms.cy;
    totalThreads = gridDim.x * blockDim.x;
    ctaStart = blockDim.x * blockIdx.x;
    cc = make_cuComplex (parms.sc, 0.0f);
    conjugCs = cuConjf(parms.cs);
   
    if ((parms.incx == 0) || (parms.incy == 0)) {
        if ((blockIdx.x == 0) && (tid == 0)) {
            if ((parms.incx == 0) && (parms.incy == 0)) {
                cuComplex tw, tz;
                w = cx[0];
                z = cy[0];
                for (i = 0; i < parms.n; i++) {
                    tw = cuCaddf (cuCmulf(cc,w), cuCmulf (parms.cs, z));
                    tz = cuCsubf (cuCmulf(cc,z), cuCmulf (conjugCs, w));
                    w = tw;
                    z = tz;
                }
                cx[0] = w;
                cy[0] = z;
            } else if (parms.incx == 0) {
                int ky = (parms.incy < 0) ? ((1 - parms.n) * parms.incy) : 0;
                cuComplex temp = cx[0];
                for (i = 0; i < parms.n; i++) {
                    w = temp;
                    z = cy[ky];
                    temp  = cuCaddf (cuCmulf (cc, w), cuCmulf (parms.cs, z));
                    cy[ky]= cuCsubf (cuCmulf (cc, z), cuCmulf (conjugCs, w));
                    ky += parms.incy;
                }
                cx[0] = temp;
            } else {
                int kx = (parms.incx < 0) ? ((1 - parms.n) * parms.incx) : 0;
                cuComplex temp = cy[0];
                for (i = 0; i < parms.n; i++) {
                    w = cx[kx];
                    z = temp;
                    cx[kx] = cuCaddf (cuCmulf (cc, w), cuCmulf (parms.cs, z));
                    temp   = cuCsubf (cuCmulf (cc, z), cuCmulf (conjugCs, w));
                    kx += parms.incx;
                }
                cy[0] = temp;
            }
        }
    } else if ((parms.incx == parms.incy) && (parms.incx > 0)) {
        /* equal, positive, increments */
        if (parms.incx == 1) {
            /* both increments equal to 1 */
            for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
                w = cx[i];
                z = cy[i];
                cx[i] = cuCaddf (cuCmulf (cc, w), cuCmulf (parms.cs, z));
                cy[i] = cuCsubf (cuCmulf (cc, z), cuCmulf (conjugCs, w));
            }
        } else {
            /* equal, positive, non-unit increments. */
            for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
                w = cx[i*parms.incx];
                z = cy[i*parms.incx];
                cx[i*parms.incx]=cuCaddf (cuCmulf(cc,w), cuCmulf(parms.cs, z));
                cy[i*parms.incx]=cuCsubf (cuCmulf(cc,z), cuCmulf(conjugCs, w));
            }
        }
    } else {
        /* unequal or nonpositive increments */
        int ix = ((parms.incx < 0) ? ((1 - n) * parms.incx) : 0);
        int iy = ((parms.incy < 0) ? ((1 - n) * parms.incy) : 0);
        for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
            w = cx[ix+i*parms.incx];
            z = cy[iy+i*parms.incy];
            cx[ix+i*parms.incx] = cuCaddf (cuCmulf(cc,w), cuCmulf(parms.cs,z));
            cy[iy+i*parms.incy] = cuCsubf (cuCmulf(cc,z), cuCmulf(conjugCs,w));
        }
    }
}

