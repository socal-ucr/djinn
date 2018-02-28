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

/* This file contains the implementation of the BLAS-1 function dcopy */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "caffe/cublas-1.1/cublas11.h"   /* CUBLAS public header file  */
#include "caffe/cublas-1.1/cublasP.h"  /* CUBLAS private header file */

texture<double> texX;

__global__ void dcopy_gld_main (struct cublasDcopyParams parms);
__global__ void dcopy_tex_main (struct cublasDcopyParams parms);

/*
 * void 
 * dcopy (int n, const double *x, int incx, double *y, int incy)
 *
 * copies the single precision vector x to the single precision vector y. For 
 * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if 
 * incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar 
 * way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 * y      single precision vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * y      contains single precision vector x
 *
 * Reference: http://www.netlib.org/blas/dcopy.f
 *
 * Error status for this function can be retrieved via cublasGetError(). 
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasDcopy (int n, const double *x, int incx,
                                     double *y, int incy)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasDcopyParams params;
    cudaError_t cudaStat;
    int nbrCtas;
    int elemsPerCta;
    int threadsPerCta;
    int sizeX = n * (imax (1, abs(incx)));
    size_t texXOfs = 0;
    int useTexture;

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
    params.sx = x;
    params.incx = incx;
    params.sy = y;
    params.incy = incy;
    params.texXOfs = (int)texXOfs;

    cublasVectorSplay (n, CUBLAS_DCOPY_THREAD_MIN, CUBLAS_DCOPY_THREAD_MAX,
                       CUBLAS_DCOPY_CTAS_MAX, &nbrCtas, &elemsPerCta, 
                       &threadsPerCta);

    cudaStat = cudaGetLastError(); /* clear error status */
    dcopy_gld_main<<<nbrCtas,threadsPerCta>>>(params);
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}

__global__ void dcopy_gld_main (struct cublasDcopyParams parms)
{
#undef  USE_TEX
#define USE_TEX 0
#include "caffe/cublas-1.1/dcopy.h"
}

__global__ void dcopy_tex_main (struct cublasDcopyParams parms)
{
}
