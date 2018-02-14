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

/* This file contains the implementation of the BLAS-1 function srotg */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

/*
 * void 
 * cublasCrotg (cuComplex *ca, cuComplex *cb, float *sc, float *cs)
 *
 * constructs the complex Givens tranformation
 *
 *        ( sc  cs )
 *    G = (        ) ,  sc^2 + cabs(cs)^2 = 1,
 *        (-cs  sc )
 * 
 * which zeros the second entry of the complex 2-vector transpose(ca, cb).
 *
 * The quantity ca/cabs(ca)*norm(ca,cb) overwrites ca in storage. The 
 * function crot (n, x, incx, y, incy, sc, cs) is normally called next
 * to apply the transformation to a 2 x n matrix.
 *
 * Input
 * -----
 * ca     single-precision complex precision scalar
 * cb     single-precision complex scalar
 *
 * Output
 * ------
 * ca     single-precision complex ca/cabs(ca)*norm(ca,cb)
 * sc     single-precision cosine component of rotation matrix
 * cs     single-precision complex sine component of rotation matrix
 *
 * Reference: http://www.netlib.org/blas/crotg.f
 *
 * This function does not set any error status.
 */
__host__ void CUBLASAPI cublasCrotg (cuComplex *pca, cuComplex cb, float *psc,
                                     cuComplex *pcs)
{
    cuComplex ca = *pca;
    float sc = *psc;
    cuComplex cs = *pcs;

    if (cuCabsf(ca) == 0.0f) {
        sc = 0.0f;
        cs = make_cuComplex (1.0f, 0.0f);
        ca = cb;
    } else {
        cuComplex alpha;
        float norm, scale;
        cuComplex tempA, tempB;
        /* Use scale factor to avoid intermediate underflow / overflow */
        scale = cuCabsf(ca) + cuCabsf(cb);
        tempA.x = ca.x / scale;
        tempA.y = ca.y / scale;
        tempB.x = cb.x / scale;
        tempB.y = cb.y / scale;
        norm = scale * sqrt (cuCabsf(tempA) * cuCabsf(tempA) + 
                             cuCabsf(tempB) * cuCabsf(tempB));
        alpha = ca;
        alpha.x = alpha.x / cuCabsf(ca);
        alpha.y = alpha.y / cuCabsf(ca);
        sc = cuCabsf(ca) / norm;
        cs = cuCmulf (alpha, cuConjf(cb));
        cs.x = cs.x / norm;
        cs.y = cs.y / norm;
        ca.x = alpha.x * norm;
        ca.y = alpha.y * norm;
    }
    *pca = ca;
    *psc = sc;
    *pcs = cs;
}
