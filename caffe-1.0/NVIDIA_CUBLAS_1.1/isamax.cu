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

/* This file contains the implementation of the BLAS-1 function isamax */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

texture<float> texX;

__global__ void isamax_gld_main (struct cublasIsamaxParams parms);
__global__ void isamax_tex_main (struct cublasIsamaxParams parms);

/*
 * int 
 * isamax (int n, const float *x, int incx)
 *
 * finds the smallest index of the maximum magnitude element of single
 * precision vector x; that is, the result is the first i, i = 0 to n - 1, 
 * that maximizes abs(x[1 + i * incx])).
 * 
 * Input
 * -----
 * n      number of elements in input vector
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the smallest index (0 if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/isamax.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ int CUBLASAPI cublasIsamax (int n, const float *x, int incx)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasIsamaxParams params;
    float *devPtrTmx;
    int *devPtrTix;
    cublasStatus status;
    cudaError_t cudaStat;
    int nbrCtas;
    int threadsPerCta;
    int idx = 0;
    int *tix;
    float *tmx;
    int i, jmax;
    float smax, xabs;
    int sizeX = n * (imax (1, abs(incx)));
    size_t texXOfs = 0;
    int useTexture;

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return idx;
    }

    /* early out if nothing to do */
    if ((n <= 0) || (incx <= 0)) {
        return idx;
    }

    if (n < CUBLAS_ISAMAX_CTAS) {
         nbrCtas = n;
         threadsPerCta = CUBLAS_ISAMAX_THREAD_COUNT;
    } else {
         nbrCtas = CUBLAS_ISAMAX_CTAS;
         threadsPerCta = CUBLAS_ISAMAX_THREAD_COUNT;
    }

    useTexture = sizeX < CUBLAS_MAX_1DBUF_SIZE;

    /* Currently, the overhead for using textures is high. Do not use texture
     * for vectors that are short, or those that are aligned and have unit
     * stride and thus have nicely coalescing GLDs.
     */
    if ((n < 100000) || /* experimental bound */
        ((sizeX == n) && (!(((uintptr_t) x) % CUBLAS_WORD_ALIGN)))) {
        useTexture = 0;
    }

    if (useTexture) {
        if ((cudaStat=cudaBindTexture (&texXOfs,texX,x,sizeX*sizeof(x[0]))) !=
            cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_MAPPING_ERROR);
            return idx;
        }
        texXOfs /= sizeof(x[0]);
    }

    /* allocate memory to collect results (index, maximum), one per CTA */
    status = cublasAlloc (nbrCtas, sizeof(tmx[0]), (void**)&devPtrTmx);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasSetError (ctx, status);
        return idx;
    }
    status = cublasAlloc (nbrCtas, sizeof(tix[0]), (void**)&devPtrTix);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasFree (devPtrTmx);
        cublasSetError (ctx, status);
        return idx;
    }

    /* allocate small buffer to retrieve the per-CTA results */
    tmx = (float *) calloc (nbrCtas, sizeof(tmx[0]));
    if (!tmx) {
        cublasFree (devPtrTmx);
        cublasFree (devPtrTix);
        cublasSetError (ctx, CUBLAS_STATUS_ALLOC_FAILED);
        return idx;
    }
    tix = (int *) calloc (nbrCtas, sizeof(tix[0]));
    if (!tix) {
        cublasFree (devPtrTmx);
        cublasFree (devPtrTix);
        free (tmx);
        cublasSetError (ctx, CUBLAS_STATUS_ALLOC_FAILED);
        return idx;
    }

    memset (&params, 0, sizeof(params));
    params.n  = n;
    params.sx = x;
    params.incx = incx;
    params.resMax = devPtrTmx;
    params.resPos = devPtrTix;
    params.texXOfs = (int)texXOfs;

    cudaStat = cudaGetLastError(); /* clear error status */
    if (useTexture) {
        isamax_tex_main<<<nbrCtas,threadsPerCta>>>(params);
    } else {
        isamax_gld_main<<<nbrCtas,threadsPerCta>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
        cublasFree (devPtrTmx);
        cublasFree (devPtrTix);
        free (tmx);
        free (tix);
        return idx;
    }

    /* Get idx/max results from each CTA */
    status = cublasGetVector (nbrCtas, sizeof(tmx[0]), devPtrTmx, 1, tmx, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR);
        cublasFree (devPtrTmx);
        cublasFree (devPtrTix);
        free (tmx);
        free (tix);
        return idx;
    }
    status = cublasGetVector (nbrCtas, sizeof(tix[0]), devPtrTix, 1, tix, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR);
        cublasFree (devPtrTmx);
        cublasFree (devPtrTix);
        free (tmx);
        free (tix);
        return idx;
    }

    /* find smallest index of maximum value among CTA results */
    smax = 0.0f;
    for (i = 0; i < nbrCtas; i++) {
        xabs = tmx[i];
        jmax = tix[i];
        if ((xabs > smax) || ((xabs == smax) && (jmax < idx))) {
            idx = jmax;
            smax = xabs;
        }
    }
    
    /* translate result from 0-indexed to 1-indexed */
    idx++;

    status = cublasFree (devPtrTmx);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR); /* corruption ? */
    }
    status = cublasFree (devPtrTix);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR); /* corruption ? */
    }
    free (tmx);
    free (tix);

    if (useTexture) {
        if ((cudaStat = cudaUnbindTexture (texX)) != cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR);
        }
    }

    return idx;   
}

__shared__ int   partialIMax[CUBLAS_ISAMAX_THREAD_COUNT];
__shared__ float partialSMax[CUBLAS_ISAMAX_THREAD_COUNT];

__global__ void isamax_gld_main (struct cublasIsamaxParams parms)
{
#undef  USE_TEX
#define USE_TEX 0
#include "isamax.h"
}

__global__ void isamax_tex_main (struct cublasIsamaxParams parms)
{
#undef  USE_TEX
#define USE_TEX 1
#include "isamax.h"
}
