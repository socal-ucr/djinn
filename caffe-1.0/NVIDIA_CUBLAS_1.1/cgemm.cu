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

/* This file contains the implementation of the BLAS-3 function cgemm */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

// dimension m, counter i
// dimension n, counter j
// dimension k, counter l

#if (CUBLAS_CGEMM_GRIDW!=CUBLAS_CGEMM_GRIDH)
#error super tile is not square!
#endif

/* Use square 16x16 tiles to access and cache portions of source matrices A,B 
 * and result matrix C
 */
#define TILE_DIM_LOG    (4)
#define TILE_DIM        (1 << TILE_DIM_LOG)
#define TILE_SIZE       (TILE_DIM*TILE_DIM)
#define SUP_TILE_DIM    (TILE_DIM*CUBLAS_CGEMM_GRIDW)

/* In cases where there are more tile elements than threads in a CTA, each
 * thread needs to walk through the tile. To keep the walking pattern simple,
 * we make sure that the number of threads is an integral multiple of the
 * number of elements (i.e. each thread deals with exactly the same number
 * of elements), and that tile dimension (number of rows / number of columns)
 * divides the thread count without remainder. After assigning an initial
 * element to each thread, the thread can then access further elements by 
 * remaining in the same tile row and merely stepping through columns that
 * are COL_INCR apart.
 */
#if ((TILE_SIZE%CUBLAS_CGEMM_THREAD_COUNT)!=0)
#error TILE_SIZE and THREAD_COUNT do not divide evenly!
#endif
#if ((CUBLAS_CGEMM_THREAD_COUNT%TILE_DIM)!=0)
#error THREAD_COUNT and TILE_DIM do not divide evenly!
#endif

#define COL_INCR               (CUBLAS_CGEMM_THREAD_COUNT/TILE_DIM)
#define C_ELEMS_PER_THREAD     (TILE_SIZE/CUBLAS_CGEMM_THREAD_COUNT)
#define A_ELEMS_PER_THREAD     (TILE_SIZE/CUBLAS_CGEMM_THREAD_COUNT)
#define B_ELEMS_PER_THREAD     (TILE_SIZE/CUBLAS_CGEMM_THREAD_COUNT)

__global__ void cgemm_1_main_sw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_2_main_sw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_3_main_sw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_4_main_sw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_5_main_sw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_6_main_sw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_7_main_sw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_8_main_sw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_9_main_sw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_1_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_2_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_3_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_4_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_5_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_6_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_7_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_8_main_sw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_9_main_sw_gld_fulltile (struct cublasCgemmParams parms);

__global__ void cgemm_1_main_hw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_2_main_hw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_3_main_hw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_4_main_hw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_5_main_hw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_6_main_hw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_7_main_hw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_8_main_hw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_9_main_hw_gld (struct cublasCgemmParams parms);
__global__ void cgemm_1_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_2_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_3_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_4_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_5_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_6_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_7_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_8_main_hw_gld_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_9_main_hw_gld_fulltile (struct cublasCgemmParams parms);

__global__ void cgemm_1_main_sw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_2_main_sw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_3_main_sw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_4_main_sw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_5_main_sw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_6_main_sw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_7_main_sw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_8_main_sw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_9_main_sw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_1_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_2_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_3_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_4_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_5_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_6_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_7_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_8_main_sw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_9_main_sw_tex_fulltile (struct cublasCgemmParams parms);

__global__ void cgemm_1_main_hw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_2_main_hw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_3_main_hw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_4_main_hw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_5_main_hw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_6_main_hw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_7_main_hw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_8_main_hw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_9_main_hw_tex (struct cublasCgemmParams parms);
__global__ void cgemm_1_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_2_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_3_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_4_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_5_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_6_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_7_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_8_main_hw_tex_fulltile (struct cublasCgemmParams parms);
__global__ void cgemm_9_main_hw_tex_fulltile (struct cublasCgemmParams parms);

texture<float2> texA;
texture<float2> texB;

__shared__ float AA_r[(TILE_DIM+1)*TILE_DIM]; 
__shared__ float BB_r[(TILE_DIM+1)*TILE_DIM]; 
__shared__ float AA_i[(TILE_DIM+1)*TILE_DIM]; 
__shared__ float BB_i[(TILE_DIM+1)*TILE_DIM];

typedef void (*pf) (struct cublasCgemmParams parms);

static pf cgemm_hw[64] = {
    cgemm_9_main_hw_gld, /* C = alpha*transpose(A)*transpose(B) + beta*C */
    cgemm_8_main_hw_gld, /* C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
    cgemm_7_main_hw_gld, /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
    cgemm_6_main_hw_gld, /* C = alpha*conjg(transpose(A))*conjg(transpose(B))+beta*C */
    cgemm_3_main_hw_gld, /* C = alpha*transpose(A)*B + beta*C */
    cgemm_3_main_hw_gld, /* C = alpha*transpose(A)*B + beta*C */
    cgemm_2_main_hw_gld, /* C = alpha*conj(transpose(A))*B + beta*C */
    cgemm_2_main_hw_gld, /* C = alpha*conj(transpose(A))*B + beta*C */
    cgemm_5_main_hw_gld, /* C = alpha*A*transpose(B) + beta*C */
    cgemm_4_main_hw_gld, /* C = alpha*A*conj(transpose(B)) + beta*C */
    cgemm_5_main_hw_gld, /* C = alpha*A*transpose(B) + beta*C */
    cgemm_4_main_hw_gld, /* C = alpha*A*conj(transpose(B)) + beta*C */
    cgemm_1_main_hw_gld, /* C = alpha*A*B + beta*C */
    cgemm_1_main_hw_gld, /* C = alpha*A*B + beta*C */
    cgemm_1_main_hw_gld, /* C = alpha*A*B + beta*C */
    cgemm_1_main_hw_gld, /* C = alpha*A*B + beta*C */
    cgemm_9_main_hw_gld_fulltile,
    cgemm_8_main_hw_gld_fulltile,
    cgemm_7_main_hw_gld_fulltile,
    cgemm_6_main_hw_gld_fulltile,
    cgemm_3_main_hw_gld_fulltile,
    cgemm_3_main_hw_gld_fulltile,
    cgemm_2_main_hw_gld_fulltile,
    cgemm_2_main_hw_gld_fulltile,
    cgemm_5_main_hw_gld_fulltile,
    cgemm_4_main_hw_gld_fulltile,
    cgemm_5_main_hw_gld_fulltile,
    cgemm_4_main_hw_gld_fulltile,
    cgemm_1_main_hw_gld_fulltile,
    cgemm_1_main_hw_gld_fulltile,
    cgemm_1_main_hw_gld_fulltile,
    cgemm_1_main_hw_gld_fulltile,
    cgemm_9_main_hw_tex,
    cgemm_8_main_hw_tex,
    cgemm_7_main_hw_tex,
    cgemm_6_main_hw_tex,
    cgemm_3_main_hw_tex,
    cgemm_3_main_hw_tex,
    cgemm_2_main_hw_tex,
    cgemm_2_main_hw_tex,
    cgemm_5_main_hw_tex,
    cgemm_4_main_hw_tex,
    cgemm_5_main_hw_tex,
    cgemm_4_main_hw_tex,
    cgemm_1_main_hw_tex,
    cgemm_1_main_hw_tex,
    cgemm_1_main_hw_tex,
    cgemm_1_main_hw_tex,
    cgemm_9_main_hw_tex_fulltile,
    cgemm_8_main_hw_tex_fulltile,
    cgemm_7_main_hw_tex_fulltile,
    cgemm_6_main_hw_tex_fulltile,
    cgemm_3_main_hw_tex_fulltile,
    cgemm_3_main_hw_tex_fulltile,
    cgemm_2_main_hw_tex_fulltile,
    cgemm_2_main_hw_tex_fulltile,
    cgemm_5_main_hw_tex_fulltile,
    cgemm_4_main_hw_tex_fulltile,
    cgemm_5_main_hw_tex_fulltile,
    cgemm_4_main_hw_tex_fulltile,
    cgemm_1_main_hw_tex_fulltile,
    cgemm_1_main_hw_tex_fulltile,
    cgemm_1_main_hw_tex_fulltile,
    cgemm_1_main_hw_tex_fulltile
};

static pf cgemm_sw[64] = {
    cgemm_9_main_sw_gld, /* C=alpha*transpose(A)*transpose(B) + beta*C */
    cgemm_8_main_hw_gld, /* C=alpha*transpose(A)*conjg(transpose(B)) + beta*C*/
    cgemm_7_main_sw_gld, /* C=alpha*conjg(transpose(A))*transpose(B) + beta*C*/
    cgemm_6_main_sw_gld, /* C=alpha*conjg(transpose(A))*conjg(transpose(B))+beta*C */
    cgemm_3_main_sw_gld, /* C=alpha*transpose(A)*B + beta*C */
    cgemm_3_main_sw_gld, /* C=alpha*transpose(A)*B + beta*C */
    cgemm_2_main_sw_gld, /* C=alpha*conj(transpose(A))*B + beta*C */
    cgemm_2_main_sw_gld, /* C=alpha*conj(transpose(A))*B + beta*C */
    cgemm_5_main_sw_gld, /* C=alpha*A*transpose(B) + beta*C */
    cgemm_4_main_sw_gld, /* C=alpha*A*conj(transpose(B)) + beta*C */
    cgemm_5_main_sw_gld, /* C=alpha*A*transpose(B) + beta*C */
    cgemm_4_main_sw_gld, /* C=alpha*A*conj(transpose(B)) + beta*C */
    cgemm_1_main_sw_gld, /* C=alpha*A*B + beta*C */
    cgemm_1_main_sw_gld, /* C=alpha*A*B + beta*C */
    cgemm_1_main_sw_gld, /* C=alpha*A*B + beta*C */
    cgemm_1_main_sw_gld, /* C=alpha*A*B + beta*C */
    cgemm_9_main_sw_gld_fulltile,
    cgemm_8_main_sw_gld_fulltile,
    cgemm_7_main_sw_gld_fulltile,
    cgemm_6_main_sw_gld_fulltile,
    cgemm_3_main_sw_gld_fulltile,
    cgemm_3_main_sw_gld_fulltile,
    cgemm_2_main_sw_gld_fulltile,
    cgemm_2_main_sw_gld_fulltile,
    cgemm_5_main_sw_gld_fulltile,
    cgemm_4_main_sw_gld_fulltile,
    cgemm_5_main_sw_gld_fulltile,
    cgemm_4_main_sw_gld_fulltile,
    cgemm_1_main_sw_gld_fulltile,
    cgemm_1_main_sw_gld_fulltile,
    cgemm_1_main_sw_gld_fulltile,
    cgemm_1_main_sw_gld_fulltile,
    cgemm_9_main_sw_tex,
    cgemm_8_main_sw_tex,
    cgemm_7_main_sw_tex,
    cgemm_6_main_sw_tex,
    cgemm_3_main_sw_tex,
    cgemm_3_main_sw_tex,
    cgemm_2_main_sw_tex,
    cgemm_2_main_sw_tex,
    cgemm_5_main_sw_tex,
    cgemm_4_main_sw_tex,
    cgemm_5_main_sw_tex,
    cgemm_4_main_sw_tex,
    cgemm_1_main_sw_tex,
    cgemm_1_main_sw_tex,
    cgemm_1_main_sw_tex,
    cgemm_1_main_sw_tex,
    cgemm_9_main_sw_tex_fulltile,
    cgemm_8_main_sw_tex_fulltile,
    cgemm_7_main_sw_tex_fulltile,
    cgemm_6_main_sw_tex_fulltile,
    cgemm_3_main_sw_tex_fulltile,
    cgemm_3_main_sw_tex_fulltile,
    cgemm_2_main_sw_tex_fulltile,
    cgemm_2_main_sw_tex_fulltile,
    cgemm_5_main_sw_tex_fulltile,
    cgemm_4_main_sw_tex_fulltile,
    cgemm_5_main_sw_tex_fulltile,
    cgemm_4_main_sw_tex_fulltile,
    cgemm_1_main_sw_tex_fulltile,
    cgemm_1_main_sw_tex_fulltile,
    cgemm_1_main_sw_tex_fulltile,
    cgemm_1_main_sw_tex_fulltile
};

/*
 * void cublasCgemm (char transa, char transb, int m, int n, int k, 
 *                   cuComplex alpha, const cuComplex *A, int lda, 
 *                   const cuComplex *B, int ldb, cuComplex beta, 
 *                   cuComplex *C, int ldc)
 *
 * cgemm performs one of the matrix-matrix operations
 *
 *    C = alpha * op(A) * op(B) + beta*C,
 *
 * where op(X) is one of
 *
 *    op(X) = X   or   op(X) = transpose  or  op(X) = conjg(transpose(X))
 *
 * alpha and beta are single-complex scalars, and A, B and C are matrices
 * consisting of single-complex elements, with op(A) an m x k matrix, op(B)
 * a k x n matrix and C an m x n matrix.
 *
 * Input
 * -----
 * transa specifies op(A). If transa == 'N' or 'n', op(A) = A. If transa == 
 *        'T' or 't', op(A) = transpose(A). If transa == 'C' or 'c', op(A) = 
 *        conjg(transpose(A)).
 * transb specifies op(B). If transa == 'N' or 'n', op(B) = B. If transb == 
 *        'T' or 't', op(B) = transpose(B). If transb == 'C' or 'c', op(B) = 
 *        conjg(transpose(B)).
 * m      number of rows of matrix op(A) and rows of matrix C. It must be at
 *        least zero.
 * n      number of columns of matrix op(B) and number of columns of C. It 
 *        must be at least zero.
 * k      number of columns of matrix op(A) and number of rows of op(B). It 
 *        must be at least zero.
 * alpha  single-complex scalar multiplier applied to op(A)op(B)
 * A      single-complex array of dimensions (lda, k) if transa ==  'N' or 
 *        'n'), and of dimensions (lda, m) otherwise.
 * lda    leading dimension of A. When transa == 'N' or 'n', it must be at 
 *        least max(1, m) and at least max(1, k) otherwise.
 * B      single-complex array of dimensions (ldb, n) if transb == 'N' or 'n', 
 *        and of dimensions (ldb, k) otherwise
 * ldb    leading dimension of B. When transb == 'N' or 'n', it must be at 
 *        least max(1, k) and at least max(1, n) otherwise.
 * beta   single-complex scalar multiplier applied to C. If beta is zero, C 
 *        does not have to be a valid input.
 * C      single precision array of dimensions (ldc, n)
 * ldc    leading dimension of C. Must be at least max(1, m).
 *
 * Output
 * ------
 * C      updated according to C = alpha*op(A)*op(B) + beta*C
 *
 * Reference: http://www.netlib.org/blas/cgemm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if any of m, n, or k are < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasCgemm (char transa, char transb, int m, int n,
                                     int k, cuComplex alpha, 
                                     const cuComplex *A, int lda, 
                                     const cuComplex *B, int ldb,
                                     cuComplex beta, cuComplex *C, int ldc)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasCgemmParams params;
    cudaError_t cudaStat;
    size_t texAOfs = 0;
    size_t texBOfs = 0;
    int nrowa, nrowb;
    int notransa, notransb;
    int conja, conjb;
    int info = 0;
    int usePureHwStepper;
    int fullTilesOnly;
    int useFastImul;
    int useTexture;
    int funcIdx;
    int sizeA, sizeB;

    dim3 ctaDimsHw (((n+TILE_DIM-1)/TILE_DIM), ((m+TILE_DIM-1)/TILE_DIM));
    dim3 ctaDimsSw (CUBLAS_CGEMM_GRIDW, CUBLAS_CGEMM_GRIDH);

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }

    nrowa = (toupper(transa) == 'N') ? m : k;
    nrowb = (toupper(transb) == 'N') ? k : n;

    if ((toupper(transa) != 'N') && 
        (toupper(transa) != 'C') && 
        (toupper(transa) != 'T')) {
        info = 1;
    } 
    else if ((toupper(transb) != 'N') && 
             (toupper(transb) != 'C') && 
             (toupper(transb) != 'T')) {
        info = 2;
    }
    else if (m < 0) {
        info = 3;
    }
    else if (n < 0) {
        info = 4;
    }
    else if (k < 0) {
        info = 5;
    }
    else if (lda < imax(1, nrowa)) {
        info = 8;
    }
    else if (ldb < imax(1, nrowb)) {
        info = 10;
    }
    else if (ldc < imax(1, m)) {
        info = 13;
    }
    if (info) {
        cublasXerbla ("CGEMM ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((m == 0) || (n == 0) || 
        ((((cuCrealf(alpha) == 0.0f) && (cuCimagf(alpha) == 0.0f)) || (k == 0)) 
         && ((cuCrealf(beta) == 1.0f) && (cuCimagf(beta) == 0.0f)))) {
        return;
    }

    /* choose version using 24-bit multiplies if all dimensions are less than
     * 1410, so we can guarantee that no intra-matrix address exceeds (1410 *
     * 1410 * 8) < 2^24.
     */
    useFastImul =((lda < CUBLAS_FASTIMUL_D_MAX_DIM) && 
                  (ldb < CUBLAS_FASTIMUL_D_MAX_DIM) && 
                  (ldc < CUBLAS_FASTIMUL_D_MAX_DIM) &&
                  (m   < CUBLAS_FASTIMUL_D_MAX_DIM) && 
                  (n   < CUBLAS_FASTIMUL_D_MAX_DIM) && 
                  (k   < CUBLAS_FASTIMUL_D_MAX_DIM));

    if (useFastImul) {
        cublasFastCgemm (ctx, transa, transb, m, n, k, alpha, A, lda, B, ldb, 
                         beta, C, ldc);
        return;
    }        
   
    sizeA = lda * ((toupper(transa) == 'N') ? k : m);
    sizeB = ldb * ((toupper(transb) == 'N') ? n : k);   

    conja  = toupper(transa) == 'C';
    conjb  = toupper(transb) == 'C';
    notransa = toupper(transa) == 'N';
    notransb = toupper(transb) == 'N';

    /* We can only use texture if the matrices fit into the largest matrix 
     * size supported.
     */
    useTexture = ((sizeA < CUBLAS_MAX_1DBUF_SIZE) &&
                  (sizeB < CUBLAS_MAX_1DBUF_SIZE));

    /* choose HW-only stepping if dimensions of result matrix do not exceed the
     * maximum CTA grid dimensions.
     */
    usePureHwStepper = ((m < (CUBLAS_CTA_MAX_DIM * TILE_DIM)) &&
                        (n < (CUBLAS_CTA_MAX_DIM * TILE_DIM)));

    /* we can eliminate checking for endcases if we know all tiles are fully
     * populated. Important benchmark case!
     */
    fullTilesOnly = (((m % TILE_DIM) == 0) &&
                     ((n % TILE_DIM) == 0) &&
                     ((k % TILE_DIM) == 0));

    /* currently, texture binding is expensive, so using texture fetches
     * is a net negative for small cases. For matrices where each row is
     * aligned, GLD coalesces nicely and is faster, so don't use texture.
     */
    if ((!(((ptrdiff_t) A) % CUBLAS_WORD_ALIGN) && 
         !(((ptrdiff_t) B) % CUBLAS_WORD_ALIGN) &&
         !(lda % (CUBLAS_WORD_ALIGN / sizeof(A[0]))) &&
         !(ldb % (CUBLAS_WORD_ALIGN / sizeof(B[0]))))) {
        useTexture = 0;
    }
    
    if (useTexture){
        if ((cudaStat=cudaBindTexture (&texAOfs,texA,A,sizeA*sizeof(A[0]))) != cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_MAPPING_ERROR);
            return;
        }
        if ((cudaStat=cudaBindTexture (&texBOfs,texB,B,sizeB*sizeof(B[0]))) != cudaSuccess) {
            cudaUnbindTexture (texA);
            cublasSetError (ctx, CUBLAS_STATUS_MAPPING_ERROR);
            return;
        }
        texAOfs /= sizeof(A[0]);
        texBOfs /= sizeof(B[0]);
    }

    memset (&params, 0, sizeof(params));
    params.m = m;
    params.n = n;
    params.k = k;
    params.alpha = alpha;
    params.A = A;
    params.lda = lda;
    params.B = B;
    params.ldb = ldb;
    params.beta = beta;
    params.C = C;
    params.ldc =ldc;
    params.texAOfs = (int)texAOfs;
    params.texBOfs = (int)texBOfs;
    
    funcIdx = ((useTexture << 5) | (fullTilesOnly << 4) | (notransa << 3) | 
               (notransb << 2) | (conja << 1) | (conjb << 0));

    cudaStat = cudaGetLastError(); /* clear error status */
    if (usePureHwStepper) {
        cgemm_hw[funcIdx]<<<ctaDimsHw,CUBLAS_CGEMM_THREAD_COUNT>>>(params);
    } else {
        cgemm_sw[funcIdx]<<<ctaDimsSw,CUBLAS_CGEMM_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */
    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
    if (useTexture) {
        if ((cudaStat = cudaUnbindTexture (texA)) != cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if ((cudaStat = cudaUnbindTexture (texB)) != cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR);
        }
    }
}

__global__ void cgemm_1_main_sw_gld (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_2_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_3_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_4_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_5_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_6_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_7_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_8_main_sw_gld (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_9_main_sw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_1_main_hw_gld (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_2_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_3_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_4_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_5_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_6_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_7_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_8_main_hw_gld (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_9_main_hw_gld (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_1_main_sw_gld_fulltile (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_2_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_3_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_4_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_5_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_6_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_7_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_8_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_9_main_sw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_1_main_hw_gld_fulltile (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_2_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_3_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_4_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_5_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_6_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_7_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_8_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_9_main_hw_gld_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           0
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

/*************************************************************************************/

__global__ void cgemm_1_main_sw_tex (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_2_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_3_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_4_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_5_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_6_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_7_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_8_main_sw_tex (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_9_main_sw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_1_main_hw_tex (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_2_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_3_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_4_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_5_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_6_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_7_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_8_main_hw_tex (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_9_main_hw_tex (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   0
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_1_main_sw_tex_fulltile (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_2_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_3_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_4_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_5_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_6_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_7_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_8_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_9_main_sw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 1
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_1_main_hw_tex_fulltile (struct cublasCgemmParams parms) 
{
    /* C = alpha*A*B + beta*C. */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_2_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conj(transpose(A))*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_3_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*B + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            0
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_4_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*conj(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_5_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*A*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            0
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_6_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_7_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*conjg(transpose(A))*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            1
#define CONJGB            0
#include "cgemm.h"
}

__global__ void cgemm_8_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /*  C = alpha*transpose(A)*conjg(transpose(B)) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            1
#include "cgemm.h"
}

__global__ void cgemm_9_main_hw_tex_fulltile (struct cublasCgemmParams parms)
{
    /* C = alpha*transpose(A)*transpose(B) + beta*C */
#undef  USE_TEX
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#undef  USE_MIXED_STEPPER
#undef  TRANSA
#undef  TRANSB
#undef  CONJGA
#undef  CONJGB
#define USE_TEX           1
#define FAST_IMUL         0
#define FULL_TILES_ONLY   1
#define USE_MIXED_STEPPER 0
#define TRANSA            1
#define TRANSB            1
#define CONJGA            0
#define CONJGB            0
#include "cgemm.h"
}
