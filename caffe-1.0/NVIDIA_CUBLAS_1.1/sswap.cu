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

/* This file contains the implementation of the BLAS-1 function sswap */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

texture<float> texX;
texture<float> texY;

__global__ void sswap_gld_main (struct cublasSswapParams parms);
__global__ void sswap_tex_main (struct cublasSswapParams parms);

/*
 * void 
 * cublasSswap (int n, float *x, int incx, float *y, int incy)
 *
 * interchanges single-precision vector x with single precision vector y..
 * For i = 0 to n - 1, it interchanges x[lx + i * incx] with y[ly + i *
 * incy], where lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx, and ly 
 * is defined in a similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single-precision vector with n elements
 * incx   storage spacing between elements of x
 * y      single-precision vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * x      input vector y (unchanged if n <= 0)
 * y      input vector x (unchanged if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/sswap.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status 
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSswap (int n, float *x, int incx, float *y,
                                     int incy)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasSswapParams params;
    cudaError_t cudaStat;
    int nbrCtas;
    int elemsPerCta;
    int threadsPerCta;
    int sizeX = n * (imax (1, abs(incx)));
    int sizeY = n * (imax (1, abs(incy)));
    size_t texXOfs = 0;
    size_t texYOfs = 0;
    int useTexture;

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }
    /* early out if nothing to do */
    if (n <= 0) {
        return;
    }

    useTexture = ((sizeX < CUBLAS_MAX_1DBUF_SIZE) &&
                  (sizeY < CUBLAS_MAX_1DBUF_SIZE));

    /* Currently, the overhead for using textures is high. Do not use texture
     * for vectors that are short, or those that are aligned and have unit
     * stride and thus have nicely coalescing GLDs.
     */
    if ((n < 50000) || /* experimental bound */
        ((sizeX == n) && (sizeY == n) && 
         (!(((uintptr_t) x) % CUBLAS_WORD_ALIGN)) && 
         (!(((uintptr_t) y) % CUBLAS_WORD_ALIGN)))) {
        useTexture = 0;
    }

    if (useTexture) {
        if ((cudaStat=cudaBindTexture (&texXOfs,texX,x,sizeX*sizeof(x[0]))) !=
            cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_MAPPING_ERROR);
            return;
        }
        if ((cudaStat=cudaBindTexture (&texYOfs,texY,y,sizeY*sizeof(y[0]))) !=
            cudaSuccess) {
            cudaUnbindTexture (texX);
            cublasSetError (ctx, CUBLAS_STATUS_MAPPING_ERROR);
            return;
        }
        texXOfs /= sizeof(x[0]);
        texYOfs /= sizeof(y[0]);
    }

    memset (&params, 0, sizeof(params));
    params.n  = n;
    params.sx = x;
    params.incx = incx;
    params.sy = y;
    params.incy = incy;
    params.texXOfs = (int)texXOfs;
    params.texYOfs = (int)texYOfs;

    cublasVectorSplay (n, CUBLAS_SSWAP_THREAD_MIN, CUBLAS_SSWAP_THREAD_MAX,
                       CUBLAS_SSWAP_CTAS_MAX, &nbrCtas, &elemsPerCta, 
                       &threadsPerCta);

    cudaStat = cudaGetLastError(); /* clear error status */
    if (useTexture) {
        sswap_tex_main<<<nbrCtas,threadsPerCta>>>(params);
    } else {
        sswap_gld_main<<<nbrCtas,threadsPerCta>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }

    if (useTexture) {
        if ((cudaStat = cudaUnbindTexture (texX)) != cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if ((cudaStat = cudaUnbindTexture (texY)) != cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR);
        }
    }
}

__global__ void sswap_gld_main (struct cublasSswapParams parms)
{
#undef  USE_TEX
#define USE_TEX 0
#include "sswap.h"
}

__global__ void sswap_tex_main (struct cublasSswapParams parms)
{
#undef  USE_TEX
#define USE_TEX 1
#include "sswap.h"
}
