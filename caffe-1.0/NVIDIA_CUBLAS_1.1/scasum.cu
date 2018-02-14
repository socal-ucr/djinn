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

/* This file contains the implementation of the BLAS-1 function scasum */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

texture<float2> texX;

__global__ void scasum_gld_main (struct cublasScasumParams parms);
__global__ void scasum_tex_main (struct cublasScasumParams parms);

/*
 * float 
 * cublasScasum (int n, const cuDouble *x, int incx)
 *
 * takes the sum of the absolute values of a complex vector and returns a
 * single precision result. Note that this is not the L1 norm of the vector.
 * The result is the sum from 0 to n-1 of abs(real(x[ix+i*incx])) +
 * abs(imag(x(ix+i*incx))), where ix = 1 if incx <= 0, else ix = 1+(1-n)*incx.
 * 
 * Input
 * -----
 * n      number of elements in input vector
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the single precision sum of absolute values of real and imaginary
 * parts (0 if n <= 0 or incx <= 0, or if an error occurs)
 *
 * Reference: http://www.netlib.org/blas/scasum.f
 *
 * Error status for this function can be retrieved via cublasGetError(). 
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ float CUBLASAPI cublasScasum (int n, const cuComplex *x, int incx)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasScasumParams params;
    float *devPtrT;
    cublasStatus status;
    cudaError_t cudaStat;
    int nbrCtas;
    int threadsPerCta;
    float sum = 0.0f;
    float *tx;
    int i;
    int useTexture;
    int sizeX = n * (imax (1, abs(incx)));
    size_t texXOfs = 0;

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return sum;
    }

    /* early out if nothing to do */
    if ((n <= 0) || (incx <= 0)) {
        return sum;
    }

    if (n < CUBLAS_SCASUM_CTAS) {
         nbrCtas = n;
         threadsPerCta = CUBLAS_SCASUM_THREAD_COUNT;
    } else {
         nbrCtas = CUBLAS_SCASUM_CTAS;
         threadsPerCta = CUBLAS_SCASUM_THREAD_COUNT;
    }
    
    useTexture = sizeX < CUBLAS_MAX_1DBUF_SIZE;

    /* Currently, the overhead for using textures is high. Do not use texture
     * for vectors that are short, or those that are aligned and have unit
     * stride and thus have nicely coalescing GLDs.
     */
    if ((n < 120000) || /* experimental bound */
        ((sizeX == n) && (!(((uintptr_t) x) % CUBLAS_LONG_ALIGN)))) {
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
    params.cx = x;
    params.incx = incx;
    params.result = devPtrT;
    params.texXOfs = (int)texXOfs;

    cudaStat = cudaGetLastError(); /* clear error status */
    if (useTexture) {
        scasum_tex_main<<<nbrCtas,threadsPerCta>>>(params);
    } else {
        scasum_gld_main<<<nbrCtas,threadsPerCta>>>(params);
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
    }    

    for (i = 0; i < nbrCtas; i++) {
        sum += tx[i];
    }

    if (useTexture) {
        if ((cudaStat = cudaUnbindTexture (texX)) != cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR);
        }
    }
    status = cublasFree (devPtrT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR); /* corruption ? */
    }
    free (tx);
    return sum;   
}

__shared__ float partialSum[CUBLAS_SCASUM_THREAD_COUNT];

__global__ void scasum_gld_main (struct cublasScasumParams parms) 
{
#undef  USE_TEX
#define USE_TEX 0
#include "scasum.h"
}

__global__ void scasum_tex_main (struct cublasScasumParams parms) 
{
#undef  USE_TEX
#define USE_TEX 1
#include "scasum.h"
}
