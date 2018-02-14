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

/* This file contains the implementation of the BLAS-1 function srotmg */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

#define GAM     (4096.0f)
#define GAMSQ   ((GAM)*(GAM))
#define RGAMSQ  (1.0f/(GAMSQ))

/*
 * void 
 * cublasSrotmg (float *psd1, float *psd2, float *psx1, const float *psy1,
 *               float *sparam)
 *
 * constructs the modified Givens transformation matrix h which zeros
 * the second component of the 2-vector transpose(sqrt(sd1)*sx1,sqrt(sd2)*sy1).
 * With sparam[0] = sflag, h has one of the following forms:
 *
 *        sflag = -1.0f   sflag = 0.0f    sflag = 1.0f    sflag = -2.0f
 *
 *        (sh00  sh01)    (1.0f  sh01)    (sh00  1.0f)    (1.0f  0.0f)
 *    h = (          )    (          )    (          )    (          )
 *        (sh10  sh11)    (sh10  1.0f)    (-1.0f sh11)    (0.0f  1.0f)
 *
 * sparam[1] through sparam[4] contain sh00, sh10, sh01, sh11, 
 * respectively. Values of 1.0f, -1.0f, or 0.0f implied by the value 
 * of sflag are not stored in sparam.
 *
 * Input
 * -----
 * sd1    single precision scalar
 * sd2    single precision scalar
 * sx1    single precision scalar
 * sy1    single precision scalar
 *
 * Output
 * ------
 * sd1    changed to represent the effect of the transformation
 * sd2    changed to represent the effect of the transformation
 * sx1    changed to represent the effect of the transformation
 * sparam 5-element vector. sparam[0] is sflag described above. sparam[1] 
 *        through sparam[4] contain the 2x2 rotation matrix h: sparam[1]
 *        contains sh00, sparam[2] contains sh10, sparam[3] contains sh01,
 *        and sprams[4] contains sh11.
 *
 * Reference: http://www.netlib.org/blas/srotmg.f
 *
 * This functions does not set any error status.
 */
__host__ void CUBLASAPI cublasSrotmg (float *psd1, float *psd2, float *psx1, 
                                      const float *psy1, float *sparam)
{
    float sd1 = *psd1;
    float sd2 = *psd2;
    float sx1 = *psx1;
    float sy1 = *psy1;
    float sp1, sp2, sq1, sq2, su;
    float sh00, sh01, sh10, sh11;
    float sflag, stemp;

    if (sd1 < 0.0f) {
        sparam[0] = -1.0f; // sflag
        sparam[1] = 0.0f;
        sparam[2] = 0.0f;
        sparam[3] = 0.0f;
        sparam[4] = 0.0f;
        *psd1 = sd1;
        *psd2 = sd2;
        *psx1 = sx1;
        return;
    }

    /* sd1 nonnegative */
    sp2 = sd2 * sy1;
    if (sp2 == 0.0f) {
        sparam[0] = -2.0f; // sflag
        return;
    }
    
    /* regular case */
    sp1 = sd1 * sx1;
    sq2 = sp2 * sy1;
    sq1 = sp1 * sx1;

    if (fabs(sq1) > fabs(sq2)) {
        sh10 = -sy1 / sx1;
        sh01 =  sp2 / sp1;
        su = 1.0f - sh01 * sh10;

        if (su <= 0.0f) {
            sparam[0] = -1.0f; // sflag
            sparam[1] = 0.0f;
            sparam[2] = 0.0f;
            sparam[3] = 0.0f;
            sparam[4] = 0.0f;
            *psd1 = 0.0f;
            *psd2 = 0.0f;
            *psx1 = 0.0f;
            return;
        }
        sflag = 0.0f;
        sh00 = 1.0f;
        sh11 = 1.0f;
        sd1 = sd1 / su;
        sd2 = sd2 / su;
        sx1 = sx1 * su;
    } else {
        if (sq2 < 0.0f) {
            sparam[0] = -1.0f; // sflag
            sparam[1] = 0.0f;
            sparam[2] = 0.0f;
            sparam[3] = 0.0f;
            sparam[4] = 0.0f;
            *psd1 = 0.0f;
            *psd2 = 0.0f;
            *psx1 = 0.0f;
            return;
        }
        sflag = 1.0f;
        sh00 = sp1 / sp2;
        sh11 = sx1 / sy1;
        sh10 = -1.0f;
        sh01 =  0.0f;

        su = 1.0f + sh00 * sh11;
        stemp = sd2 / su;
        sd2 = sd1 / su;
        sd1 = stemp;
        sx1 = sy1 * su;
    }
    /* SCALE-CHECK */
    while ((sd1 <= RGAMSQ) && (sd1 != 0.0f)) {
        sflag = -1.0f;
        sd1 = sd1 * GAMSQ;
        sx1 = sx1 / GAM;
        sh00 = sh00 / GAM;
        sh01 = sh01 / GAM;
    }
    while (sd1 >= GAMSQ) {
        sflag = -1.0f;
        sd1  = sd1 / GAMSQ;
        sx1  = sx1 * GAM;
        sh10 = sh10 * GAM;
        sh01 = sh11 * GAM;
    }
    while ((fabs(sd2) <= RGAMSQ) && (sd2 != 0.0f)) {
        sflag = -1.0f;
        sd2  = sd2 * GAMSQ;
        sh10 = sh10 / GAM;
        sh11 = sh11 / GAM;
    }
    while (fabs(sd2) >= GAMSQ) {
        sflag = -1.0f;
        sd2  = sd2 / GAMSQ;
        sh10 = sh10 * GAM;
        sh11 = sh11 * GAM;
    }

    sparam[0] = sflag;
    if (sflag == -1.0) {
        sparam[1] = sh00;
        sparam[2] = sh10;
        sparam[3] = sh01;
        sparam[4] = sh11;
    } else if (sflag == 0.0) {
        sparam[2] = sh10;
        sparam[3] = sh01;
    } else if (sflag == 1.0) {
        sparam[1] = sh00;
        sparam[4] = sh11;
    }
    *psd1 = sd1;
    *psd2 = sd2;
    *psx1 = sx1;
}
