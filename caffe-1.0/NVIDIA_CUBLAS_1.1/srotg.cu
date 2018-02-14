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
 * cublasSrotg (float *sa, float *sb, float *sc, float *ss)
 *
 * constructs the Givens tranformation
 *
 *        ( sc  ss )
 *    G = (        ) ,  sc^2 + ss^2 = 1,
 *        (-ss  sc )
 *
 * which zeros the second entry of the 2-vector transpose(sa, sb).
 *
 * The quantity r = (+/-) sqrt (sa^2 + sb^2) overwrites sa in storage. The 
 * value of sb is overwritten by a value z which allows sc and ss to be 
 * recovered by the following algorithm:
 *
 *    if z=1          set sc = 0.0 and ss = 1.0
 *    if abs(z) < 1   set sc = sqrt(1-z^2) and ss = z
 *    if abs(z) > 1   set sc = 1/z and ss = sqrt(1-sc^2)
 *
 * The function srot (n, x, incx, y, incy, sc, ss) normally is called next
 * to apply the transformation to a 2 x n matrix.
 *
 * Input
 * -----
 * sa     single precision scalar
 * sb     single precision scalar
 *
 * Output
 * ------
 * sa     single precision r
 * sb     single precision z
 * sc     single precision result
 * ss     single precision result
 *
 * Reference: http://www.netlib.org/blas/srotg.f
 *
 * This function does not set any error status.
 */
__host__ void CUBLASAPI cublasSrotg (float *psa, float *psb, float *psc,
                                     float *pss)
{
    float sa = *psa;
    float sb = *psb;
    float sc = *psc;
    float ss = *pss;
    float r, u, v;

    if (fabs(sa) > fabs(sb)) {
        /* here |sa| > |sb| */
        u = sa + sa;
        v = sb / u;
     
        /* note that u and r have the sign of sa */
        r = ((float)sqrt(0.25f + v * v)) * u;

        /* note that sc is positive */
        sc = sa /r;
        ss = v * (sc + sc);
        sb = ss;
        sa = r;
    } else {
        /* here |sa| <= |sb| */
        if (sb != 0.0f) {
            u = sb + sb;
            v = sa / u;

            /* note that u and r have the sign if sb
             * (r is immediately stored in sa)
             */
            sa = ((float)sqrt (0.25f + v * v)) * u;
            
            /* note that ss is positive */
            ss = sb / sa;
            sc = v * (ss + ss);
            if (sc != 0.0f) {
                sb = 1.0f / sc;
            } else {
                sb = 1.0f;
            }
        } else {
            /* here sa = sb = 0.0 */
            sc = 1.0f;
            ss = 0.0f;
        }
    }
    *psa = sa;
    *psb = sb;
    *psc = sc;
    *pss = ss;
}
