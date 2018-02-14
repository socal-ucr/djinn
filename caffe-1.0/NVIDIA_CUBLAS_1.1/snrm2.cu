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

/* This file contains the implementation of the BLAS-1 function snrm2 */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

#define CUBLAS_SNRM2_STATE_ZERO    0
#define CUBLAS_SNRM2_STATE_TINY    1
#define CUBLAS_SNRM2_STATE_NORMAL  2
#define CUBLAS_SNRM2_STATE_HUGE    3
#define CUBLAS_SNRM2_STATE_DONE    4

texture<float> texX;

__global__ void snrm2_gld_main (struct cublasSnrm2Params parms);
__global__ void snrm2_tex_main (struct cublasSnrm2Params parms);
__host__ static float local_snrm2 (int n, const float *sx, int incx);

/*
 * float 
 * snrm2 (int n, const float *x, int incx)
 *
 * computes the Euclidean norm of the single precision n-vector x (with
 * storage increment incx). This code uses a multiphase model of 
 * accumulation to avoid intermediate underflow and overflow.
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns Euclidian norm (0 if n <= 0 or incx <= 0, or if an error occurs)
 *
 * Reference: http://www.netlib.org/blas/snrm2.f
 * Reference: http://www.netlib.org/slatec/lin/snrm2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 * 
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ float CUBLASAPI cublasSnrm2 (int n, const float *x, int incx)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasSnrm2Params params;
    float *devPtrT;
    cublasStatus status;
    cudaError_t cudaStat;
    int nbrCtas;
    int threadsPerCta;
    float sum = 0.0f;
    float *tx;
    int sizeX = n * (imax (1, abs(incx)));
    size_t texXOfs = 0;
    int useTexture;

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return sum;
    }

    /* early out if nothing to do */
    if ((n <= 0) || (incx <= 0)) {
        return sum;
    }

    if (n < CUBLAS_SNRM2_CTAS) {
         nbrCtas = n;
         threadsPerCta = CUBLAS_SNRM2_THREAD_COUNT;
    } else {
         nbrCtas = CUBLAS_SNRM2_CTAS;
         threadsPerCta = CUBLAS_SNRM2_THREAD_COUNT;
    }

    useTexture = sizeX < CUBLAS_MAX_1DBUF_SIZE;

    /* Currently, the overhead for using textures is high. Do not use texture
     * for vectors that are short, or those that are aligned and have unit
     * stride and thus have nicely coalescing GLDs.
     */
    if ((n < 130000) || /* experimental bound */
        ((sizeX == n) && (!(((uintptr_t) x) % CUBLAS_WORD_ALIGN)))) {
        useTexture = 0;
    }

    if (useTexture) {
        if ((cudaStat=cudaBindTexture (&texXOfs,texX,x,sizeX*sizeof(x[0]))) !=
            cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_MAPPING_ERROR);
            return sum;
        }
        texXOfs /= sizeof(x[0]);
    }
    
    /* allocate memory to collect results, one per CTA */
    status = cublasAlloc (nbrCtas, sizeof(tx[0]), (void**)&devPtrT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasSetError (ctx, status);
        return sum;
    }

    /* allocate small buffer to retrieve the per-CTA results */
    tx = (float *) calloc (nbrCtas, sizeof(tx[0]));
    if (!tx) {
        cublasFree (devPtrT);
        cublasSetError (ctx, CUBLAS_STATUS_ALLOC_FAILED);
        return sum;
    }

    memset (&params, 0, sizeof(params));
    params.n  = n;
    params.sx = x;
    params.incx = incx;
    params.result = devPtrT;
    params.texXOfs = (int)texXOfs;

    cudaStat = cudaGetLastError(); /* clear error status */
    if (useTexture) {
        snrm2_tex_main<<<nbrCtas,threadsPerCta>>>(params);
    } else {
        snrm2_gld_main<<<nbrCtas,threadsPerCta>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasFree (devPtrT);
        free (tx);
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
        return sum;
    }

    /* Sum the results from each CTA */
    status = cublasGetVector (nbrCtas, sizeof(tx[0]), devPtrT, 1, tx, 1);

    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasFree (devPtrT);
        free (tx);
        cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR);
        return sum;
    }    
    
    sum = local_snrm2 (nbrCtas, tx, 1);

    status = cublasFree (devPtrT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR); /* corruption ? */
    }
    free (tx);

    if (useTexture) {
        if ((cudaStat = cudaUnbindTexture (texX)) != cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR);
        }
    }

    return sum;   
}

/*
 * float snrm2 (int n, const float *sx, int incx);
 *
 * IN:  n    number of elements in input vector
 *      sx   single precision vector with n elements
 *      incx storage spacing between elements of sx
 *
 * OUT: returns Euclidian norm (0 if n <= 0 or incx <= 0)
 *
 * BLAS level 1, see http://www.netlib.org/blas/snrm2.f
 *
 * Euclidean norm of the n-vector stored in sx with storage
 * increment incx. This version uses a multi-phase model of
 * accumulation to avoid intermediate underflow and overflow.
 * See http://www.netlib.org/slatec/lin/snrm2.f
 */
__host__ static float local_snrm2 (int n, const float *sx, int incx)
{
    float cutLo = 4.441e-16f;
    float cutHi = 1.304e+19f;
    unsigned int i, ns, state;
    volatile float sum = 0.0f;
    volatile float hiTest;
    volatile float t = 0.0f;
    volatile float ta = 0.0f;
    volatile float xmax = 0.0f;
    volatile float xmaxRecip;

    if ((n <= 0) || (incx <= 0)) {
        return sum;
    }
    ns = n * incx;
    hiTest = cutHi / (float)n;
    i = 0;
    state = CUBLAS_SNRM2_STATE_ZERO;
    while (state != CUBLAS_SNRM2_STATE_DONE) {
        switch (state) {
        case CUBLAS_SNRM2_STATE_ZERO:
            while ((i < ns) && ((t = sx[i]) == 0.0f)) {
                i += incx;
            }
            if (i >= ns) {
                state = CUBLAS_SNRM2_STATE_DONE;
            } else {
                state = CUBLAS_SNRM2_STATE_TINY;
            }
            break;            
        case CUBLAS_SNRM2_STATE_TINY:
            xmax = (float)fabs(t);
            xmaxRecip = 1.0f / xmax;
            while ((i < ns) && ((ta = (float)fabs(t = sx[i])) < cutLo)) {
                if (ta > xmax) {
                    /* Adjust scale factor */
                    t = xmax / t;
                    sum = 1.0f + sum * t * t;
                    xmax = ta;
                    xmaxRecip = 1.0f / xmax;
                } else {
                    t = t * xmaxRecip;
                    sum += t * t;
                }
                i += incx; 
            }
            if (i >= ns) {
                sum = (float)sqrt(sum);
                sum = xmax * sum;
                state = CUBLAS_SNRM2_STATE_DONE;
            } else {            
                state = CUBLAS_SNRM2_STATE_NORMAL;
            }
            break;
        case CUBLAS_SNRM2_STATE_NORMAL:
            sum = (sum * xmax) * xmax;
            while ((i < ns) && ((ta = (float)fabs(t = sx[i])) < hiTest)) {
                sum += t * t;
                i += incx;  
            }
            if (i >= ns) {
                sum = (float)sqrt(sum);
                state = CUBLAS_SNRM2_STATE_DONE;
            } else {
                state = CUBLAS_SNRM2_STATE_HUGE;
            }
            break;
        case CUBLAS_SNRM2_STATE_HUGE:
            xmax = ta;
            xmaxRecip = 1.0f / xmax;
            sum = (sum * xmaxRecip) * xmaxRecip;
            while (i < ns) {
                t = sx[i];
                ta = (float)fabs(t);
                if (ta > xmax) {
                    /* Adjust scale factor */
                    t = xmax / t;
                    sum = 1.0f + sum * t * t;
                    xmax = ta;
                    xmaxRecip = 1.0f / xmax;
                } else {
                    t = t * xmaxRecip;
                    sum += t * t;
                }
                i += incx; 
            }
            sum = (float)sqrt (sum);
            sum = xmax * sum;
            state = CUBLAS_SNRM2_STATE_DONE;
            break;
        }
    }
    return sum;
}

__shared__ float partialSum[CUBLAS_SNRM2_THREAD_COUNT];

__global__ void snrm2_gld_main (struct cublasSnrm2Params parms) 
{
#undef USE_TEX
#define USE_TEX 0
#include "snrm2.h"
}

__global__ void snrm2_tex_main (struct cublasSnrm2Params parms) 
{
#undef USE_TEX
#define USE_TEX 1
#include "snrm2.h"
}
