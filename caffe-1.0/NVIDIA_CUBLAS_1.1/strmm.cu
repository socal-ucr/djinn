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

/* This file contains the implementation of the BLAS-3 function strmm */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

#define BLK_LOG             (5)
#define BLK                 (1 << BLK_LOG)
#define JINC                (BLK * CUBLAS_STRMM_CTAS)  // used by strmm_l
#define IINC                (BLK * CUBLAS_STRMM_CTAS)  // used by strmm_r

#if ((CUBLAS_STRMM_THREAD_COUNT<BLK))
#error block dimension must be >= threadcount
#endif

#if ((CUBLAS_STRMM_THREAD_COUNT%BLK)!=0)
#error threadcount and block dimensions do not divide evenly
#endif

#define A_NBR_COLS          (CUBLAS_STRMM_THREAD_COUNT/BLK)
#define B_NBR_COLS          (CUBLAS_STRMM_THREAD_COUNT/BLK)

#if (((BLK*BLK)%CUBLAS_STRMM_THREAD_COUNT)!=0)
#error blocksize of A and B not evenly divided by threadcount!
#endif

#define A_ELEMS_PER_THREAD  ((BLK * BLK) / CUBLAS_STRMM_THREAD_COUNT)
#define B_ELEMS_PER_THREAD  ((BLK * BLK) / CUBLAS_STRMM_THREAD_COUNT)

__global__ void strmm_r_lo_tr_main_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_unit_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_unit_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_unit_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_unit_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_unit_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_unit_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_unit_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_unit_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_unit_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_unit_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_unit_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_unit_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_unit_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_unit_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_unit_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_unit_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms);

__global__ void strmm_r_lo_tr_main_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_unit_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_unit_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_unit_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_unit_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_unit_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_unit_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_unit_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_unit_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_unit_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_unit_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_unit_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_unit_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_unit_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_unit_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_unit_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_unit_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_lo_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_r_up_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_lo_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void strmm_l_up_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_lo_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_r_up_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_lo_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);
__global__ void fast_strmm_l_up_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms);


typedef void (*pf) (struct cublasStrmmParams parms);

static pf strmm_sw[128] = {
    strmm_r_lo_tr_main_sw,
    strmm_r_up_tr_main_sw,
    strmm_r_lo_nt_main_sw,
    strmm_r_up_nt_main_sw,
    strmm_l_lo_tr_main_sw,
    strmm_l_up_tr_main_sw,
    strmm_l_lo_nt_main_sw,
    strmm_l_up_nt_main_sw,
    strmm_r_lo_tr_main_unit_sw,
    strmm_r_up_tr_main_unit_sw,
    strmm_r_lo_nt_main_unit_sw,
    strmm_r_up_nt_main_unit_sw,
    strmm_l_lo_tr_main_unit_sw,
    strmm_l_up_tr_main_unit_sw,
    strmm_l_lo_nt_main_unit_sw,
    strmm_l_up_nt_main_unit_sw,
    strmm_r_lo_tr_main_alpha0_sw,
    strmm_r_up_tr_main_alpha0_sw,
    strmm_r_lo_nt_main_alpha0_sw,
    strmm_r_up_nt_main_alpha0_sw,
    strmm_l_lo_tr_main_alpha0_sw,
    strmm_l_up_tr_main_alpha0_sw,
    strmm_l_lo_nt_main_alpha0_sw,
    strmm_l_up_nt_main_alpha0_sw,
    strmm_r_lo_tr_main_unit_alpha0_sw,
    strmm_r_up_tr_main_unit_alpha0_sw,
    strmm_r_lo_nt_main_unit_alpha0_sw,
    strmm_r_up_nt_main_unit_alpha0_sw,
    strmm_l_lo_tr_main_unit_alpha0_sw,
    strmm_l_up_tr_main_unit_alpha0_sw,
    strmm_l_lo_nt_main_unit_alpha0_sw,
    strmm_l_up_nt_main_unit_alpha0_sw,
    fast_strmm_r_lo_tr_main_sw,
    fast_strmm_r_up_tr_main_sw,
    fast_strmm_r_lo_nt_main_sw,
    fast_strmm_r_up_nt_main_sw,
    fast_strmm_l_lo_tr_main_sw,
    fast_strmm_l_up_tr_main_sw,
    fast_strmm_l_lo_nt_main_sw,
    fast_strmm_l_up_nt_main_sw,
    fast_strmm_r_lo_tr_main_unit_sw,
    fast_strmm_r_up_tr_main_unit_sw,
    fast_strmm_r_lo_nt_main_unit_sw,
    fast_strmm_r_up_nt_main_unit_sw,
    fast_strmm_l_lo_tr_main_unit_sw,
    fast_strmm_l_up_tr_main_unit_sw,
    fast_strmm_l_lo_nt_main_unit_sw,
    fast_strmm_l_up_nt_main_unit_sw,
    fast_strmm_r_lo_tr_main_alpha0_sw,
    fast_strmm_r_up_tr_main_alpha0_sw,
    fast_strmm_r_lo_nt_main_alpha0_sw,
    fast_strmm_r_up_nt_main_alpha0_sw,
    fast_strmm_l_lo_tr_main_alpha0_sw,
    fast_strmm_l_up_tr_main_alpha0_sw,
    fast_strmm_l_lo_nt_main_alpha0_sw,
    fast_strmm_l_up_nt_main_alpha0_sw,
    fast_strmm_r_lo_tr_main_unit_alpha0_sw,
    fast_strmm_r_up_tr_main_unit_alpha0_sw,
    fast_strmm_r_lo_nt_main_unit_alpha0_sw,
    fast_strmm_r_up_nt_main_unit_alpha0_sw,
    fast_strmm_l_lo_tr_main_unit_alpha0_sw,
    fast_strmm_l_up_tr_main_unit_alpha0_sw,
    fast_strmm_l_lo_nt_main_unit_alpha0_sw,
    fast_strmm_l_up_nt_main_unit_alpha0_sw,
    strmm_r_lo_tr_main_fulltile_sw,
    strmm_r_up_tr_main_fulltile_sw,
    strmm_r_lo_nt_main_fulltile_sw,
    strmm_r_up_nt_main_fulltile_sw,
    strmm_l_lo_tr_main_fulltile_sw,
    strmm_l_up_tr_main_fulltile_sw,
    strmm_l_lo_nt_main_fulltile_sw,
    strmm_l_up_nt_main_fulltile_sw,
    strmm_r_lo_tr_main_unit_fulltile_sw,
    strmm_r_up_tr_main_unit_fulltile_sw,
    strmm_r_lo_nt_main_unit_fulltile_sw,
    strmm_r_up_nt_main_unit_fulltile_sw,
    strmm_l_lo_tr_main_unit_fulltile_sw,
    strmm_l_up_tr_main_unit_fulltile_sw,
    strmm_l_lo_nt_main_unit_fulltile_sw,
    strmm_l_up_nt_main_unit_fulltile_sw,
    strmm_r_lo_tr_main_alpha0_fulltile_sw,
    strmm_r_up_tr_main_alpha0_fulltile_sw,
    strmm_r_lo_nt_main_alpha0_fulltile_sw,
    strmm_r_up_nt_main_alpha0_fulltile_sw,
    strmm_l_lo_tr_main_alpha0_fulltile_sw,
    strmm_l_up_tr_main_alpha0_fulltile_sw,
    strmm_l_lo_nt_main_alpha0_fulltile_sw,
    strmm_l_up_nt_main_alpha0_fulltile_sw,
    strmm_r_lo_tr_main_unit_alpha0_fulltile_sw,
    strmm_r_up_tr_main_unit_alpha0_fulltile_sw,
    strmm_r_lo_nt_main_unit_alpha0_fulltile_sw,
    strmm_r_up_nt_main_unit_alpha0_fulltile_sw,
    strmm_l_lo_tr_main_unit_alpha0_fulltile_sw,
    strmm_l_up_tr_main_unit_alpha0_fulltile_sw,
    strmm_l_lo_nt_main_unit_alpha0_fulltile_sw,
    strmm_l_up_nt_main_unit_alpha0_fulltile_sw,
    fast_strmm_r_lo_tr_main_fulltile_sw,
    fast_strmm_r_up_tr_main_fulltile_sw,
    fast_strmm_r_lo_nt_main_fulltile_sw,
    fast_strmm_r_up_nt_main_fulltile_sw,
    fast_strmm_l_lo_tr_main_fulltile_sw,
    fast_strmm_l_up_tr_main_fulltile_sw,
    fast_strmm_l_lo_nt_main_fulltile_sw,
    fast_strmm_l_up_nt_main_fulltile_sw,
    fast_strmm_r_lo_tr_main_unit_fulltile_sw,
    fast_strmm_r_up_tr_main_unit_fulltile_sw,
    fast_strmm_r_lo_nt_main_unit_fulltile_sw,
    fast_strmm_r_up_nt_main_unit_fulltile_sw,
    fast_strmm_l_lo_tr_main_unit_fulltile_sw,
    fast_strmm_l_up_tr_main_unit_fulltile_sw,
    fast_strmm_l_lo_nt_main_unit_fulltile_sw,
    fast_strmm_l_up_nt_main_unit_fulltile_sw,
    fast_strmm_r_lo_tr_main_alpha0_fulltile_sw,
    fast_strmm_r_up_tr_main_alpha0_fulltile_sw,
    fast_strmm_r_lo_nt_main_alpha0_fulltile_sw,
    fast_strmm_r_up_nt_main_alpha0_fulltile_sw,
    fast_strmm_l_lo_tr_main_alpha0_fulltile_sw,
    fast_strmm_l_up_tr_main_alpha0_fulltile_sw,
    fast_strmm_l_lo_nt_main_alpha0_fulltile_sw,
    fast_strmm_l_up_nt_main_alpha0_fulltile_sw,
    fast_strmm_r_lo_tr_main_unit_alpha0_fulltile_sw,
    fast_strmm_r_up_tr_main_unit_alpha0_fulltile_sw,
    fast_strmm_r_lo_nt_main_unit_alpha0_fulltile_sw,
    fast_strmm_r_up_nt_main_unit_alpha0_fulltile_sw,
    fast_strmm_l_lo_tr_main_unit_alpha0_fulltile_sw,
    fast_strmm_l_up_tr_main_unit_alpha0_fulltile_sw,
    fast_strmm_l_lo_nt_main_unit_alpha0_fulltile_sw,
    fast_strmm_l_up_nt_main_unit_alpha0_fulltile_sw
};

static pf strmm_hw[128] = {
    strmm_r_lo_tr_main_hw,
    strmm_r_up_tr_main_hw,
    strmm_r_lo_nt_main_hw,
    strmm_r_up_nt_main_hw,
    strmm_l_lo_tr_main_hw,
    strmm_l_up_tr_main_hw,
    strmm_l_lo_nt_main_hw,
    strmm_l_up_nt_main_hw,
    strmm_r_lo_tr_main_unit_hw,
    strmm_r_up_tr_main_unit_hw,
    strmm_r_lo_nt_main_unit_hw,
    strmm_r_up_nt_main_unit_hw,
    strmm_l_lo_tr_main_unit_hw,
    strmm_l_up_tr_main_unit_hw,
    strmm_l_lo_nt_main_unit_hw,
    strmm_l_up_nt_main_unit_hw,
    strmm_r_lo_tr_main_alpha0_hw,
    strmm_r_up_tr_main_alpha0_hw,
    strmm_r_lo_nt_main_alpha0_hw,
    strmm_r_up_nt_main_alpha0_hw,
    strmm_l_lo_tr_main_alpha0_hw,
    strmm_l_up_tr_main_alpha0_hw,
    strmm_l_lo_nt_main_alpha0_hw,
    strmm_l_up_nt_main_alpha0_hw,
    strmm_r_lo_tr_main_unit_alpha0_hw,
    strmm_r_up_tr_main_unit_alpha0_hw,
    strmm_r_lo_nt_main_unit_alpha0_hw,
    strmm_r_up_nt_main_unit_alpha0_hw,
    strmm_l_lo_tr_main_unit_alpha0_hw,
    strmm_l_up_tr_main_unit_alpha0_hw,
    strmm_l_lo_nt_main_unit_alpha0_hw,
    strmm_l_up_nt_main_unit_alpha0_hw,
    fast_strmm_r_lo_tr_main_hw,
    fast_strmm_r_up_tr_main_hw,
    fast_strmm_r_lo_nt_main_hw,
    fast_strmm_r_up_nt_main_hw,
    fast_strmm_l_lo_tr_main_hw,
    fast_strmm_l_up_tr_main_hw,
    fast_strmm_l_lo_nt_main_hw,
    fast_strmm_l_up_nt_main_hw,
    fast_strmm_r_lo_tr_main_unit_hw,
    fast_strmm_r_up_tr_main_unit_hw,
    fast_strmm_r_lo_nt_main_unit_hw,
    fast_strmm_r_up_nt_main_unit_hw,
    fast_strmm_l_lo_tr_main_unit_hw,
    fast_strmm_l_up_tr_main_unit_hw,
    fast_strmm_l_lo_nt_main_unit_hw,
    fast_strmm_l_up_nt_main_unit_hw,
    fast_strmm_r_lo_tr_main_alpha0_hw,
    fast_strmm_r_up_tr_main_alpha0_hw,
    fast_strmm_r_lo_nt_main_alpha0_hw,
    fast_strmm_r_up_nt_main_alpha0_hw,
    fast_strmm_l_lo_tr_main_alpha0_hw,
    fast_strmm_l_up_tr_main_alpha0_hw,
    fast_strmm_l_lo_nt_main_alpha0_hw,
    fast_strmm_l_up_nt_main_alpha0_hw,
    fast_strmm_r_lo_tr_main_unit_alpha0_hw,
    fast_strmm_r_up_tr_main_unit_alpha0_hw,
    fast_strmm_r_lo_nt_main_unit_alpha0_hw,
    fast_strmm_r_up_nt_main_unit_alpha0_hw,
    fast_strmm_l_lo_tr_main_unit_alpha0_hw,
    fast_strmm_l_up_tr_main_unit_alpha0_hw,
    fast_strmm_l_lo_nt_main_unit_alpha0_hw,
    fast_strmm_l_up_nt_main_unit_alpha0_hw,
    strmm_r_lo_tr_main_fulltile_hw,
    strmm_r_up_tr_main_fulltile_hw,
    strmm_r_lo_nt_main_fulltile_hw,
    strmm_r_up_nt_main_fulltile_hw,
    strmm_l_lo_tr_main_fulltile_hw,
    strmm_l_up_tr_main_fulltile_hw,
    strmm_l_lo_nt_main_fulltile_hw,
    strmm_l_up_nt_main_fulltile_hw,
    strmm_r_lo_tr_main_unit_fulltile_hw,
    strmm_r_up_tr_main_unit_fulltile_hw,
    strmm_r_lo_nt_main_unit_fulltile_hw,
    strmm_r_up_nt_main_unit_fulltile_hw,
    strmm_l_lo_tr_main_unit_fulltile_hw,
    strmm_l_up_tr_main_unit_fulltile_hw,
    strmm_l_lo_nt_main_unit_fulltile_hw,
    strmm_l_up_nt_main_unit_fulltile_hw,
    strmm_r_lo_tr_main_alpha0_fulltile_hw,
    strmm_r_up_tr_main_alpha0_fulltile_hw,
    strmm_r_lo_nt_main_alpha0_fulltile_hw,
    strmm_r_up_nt_main_alpha0_fulltile_hw,
    strmm_l_lo_tr_main_alpha0_fulltile_hw,
    strmm_l_up_tr_main_alpha0_fulltile_hw,
    strmm_l_lo_nt_main_alpha0_fulltile_hw,
    strmm_l_up_nt_main_alpha0_fulltile_hw,
    strmm_r_lo_tr_main_unit_alpha0_fulltile_hw,
    strmm_r_up_tr_main_unit_alpha0_fulltile_hw,
    strmm_r_lo_nt_main_unit_alpha0_fulltile_hw,
    strmm_r_up_nt_main_unit_alpha0_fulltile_hw,
    strmm_l_lo_tr_main_unit_alpha0_fulltile_hw,
    strmm_l_up_tr_main_unit_alpha0_fulltile_hw,
    strmm_l_lo_nt_main_unit_alpha0_fulltile_hw,
    strmm_l_up_nt_main_unit_alpha0_fulltile_hw,
    fast_strmm_r_lo_tr_main_fulltile_hw,
    fast_strmm_r_up_tr_main_fulltile_hw,
    fast_strmm_r_lo_nt_main_fulltile_hw,
    fast_strmm_r_up_nt_main_fulltile_hw,
    fast_strmm_l_lo_tr_main_fulltile_hw,
    fast_strmm_l_up_tr_main_fulltile_hw,
    fast_strmm_l_lo_nt_main_fulltile_hw,
    fast_strmm_l_up_nt_main_fulltile_hw,
    fast_strmm_r_lo_tr_main_unit_fulltile_hw,
    fast_strmm_r_up_tr_main_unit_fulltile_hw,
    fast_strmm_r_lo_nt_main_unit_fulltile_hw,
    fast_strmm_r_up_nt_main_unit_fulltile_hw,
    fast_strmm_l_lo_tr_main_unit_fulltile_hw,
    fast_strmm_l_up_tr_main_unit_fulltile_hw,
    fast_strmm_l_lo_nt_main_unit_fulltile_hw,
    fast_strmm_l_up_nt_main_unit_fulltile_hw,
    fast_strmm_r_lo_tr_main_alpha0_fulltile_hw,
    fast_strmm_r_up_tr_main_alpha0_fulltile_hw,
    fast_strmm_r_lo_nt_main_alpha0_fulltile_hw,
    fast_strmm_r_up_nt_main_alpha0_fulltile_hw,
    fast_strmm_l_lo_tr_main_alpha0_fulltile_hw,
    fast_strmm_l_up_tr_main_alpha0_fulltile_hw,
    fast_strmm_l_lo_nt_main_alpha0_fulltile_hw,
    fast_strmm_l_up_nt_main_alpha0_fulltile_hw,
    fast_strmm_r_lo_tr_main_unit_alpha0_fulltile_hw,
    fast_strmm_r_up_tr_main_unit_alpha0_fulltile_hw,
    fast_strmm_r_lo_nt_main_unit_alpha0_fulltile_hw,
    fast_strmm_r_up_nt_main_unit_alpha0_fulltile_hw,
    fast_strmm_l_lo_tr_main_unit_alpha0_fulltile_hw,
    fast_strmm_l_up_tr_main_unit_alpha0_fulltile_hw,
    fast_strmm_l_lo_nt_main_unit_alpha0_fulltile_hw,
    fast_strmm_l_up_nt_main_unit_alpha0_fulltile_hw
};

/*
 * void 
 * cublasStrmm (char side, char uplo, char transa, char diag, int m, int n, 
 *              float alpha, const float *A, int lda, const float *B, int ldb)
 *
 * performs one of the matrix-matrix operations
 *
 *   B = alpha * op(A) * B,  or  B = alpha * B * op(A)
 *
 * where alpha is a single-precision scalar, B is an m x n matrix composed
 * of single precision elements, and A is a unit or non-unit, upper or lower, 
 * triangular matrix composed of single precision elements. op(A) is one of
 *
 *   op(A) = A  or  op(A) = transpose(A)
 *
 * Matrices A and B are stored in column major format, and lda and ldb are 
 * the leading dimensions of the two-dimensonials arrays that contain A and 
 * B, respectively.
 *
 * Input
 * -----
 * side   specifies whether op(A) multiplies B from the left or right.
 *        If side = 'L' or 'l', then B = alpha * op(A) * B. If side =
 *        'R' or 'r', then B = alpha * B * op(A).
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo = 'U' or 'u', A is an upper triangular matrix.
 *        If uplo = 'L' or 'l', A is a lower triangular matrix.
 * transa specifies the form of op(A) to be used in the matrix 
 *        multiplication. If transa = 'N' or 'n', then op(A) = A. If
 *        transa = 'T', 't', 'C', or 'c', then op(A) = transpose(A).
 * diag   specifies whether or not A is unit triangular. If diag = 'U'
 *        or 'u', A is assumed to be unit triangular. If diag = 'N' or
 *        'n', A is not assumed to be unit triangular.
 * m      the number of rows of matrix B. m must be at least zero.
 * n      the number of columns of matrix B. n must be at least zero.
 * alpha  single precision scalar multiplier applied to op(A)*B, or
 *        B*op(A), respectively. If alpha is zero no accesses are made
 *        to matrix A, and no read accesses are made to matrix B.
 * A      single precision array of dimensions (lda, k). k = m if side =
 *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
 *        the leading k x k upper triangular part of the array A must
 *        contain the upper triangular matrix, and the strictly lower
 *        triangular part of A is not referenced. If uplo = 'L' or 'l'
 *        the leading k x k lower triangular part of the array A must
 *        contain the lower triangular matrix, and the strictly upper
 *        triangular part of A is not referenced. When diag = 'U' or 'u'
 *        the diagonal elements of A are no referenced and are assumed
 *        to be unity.
 * lda    leading dimension of A. When side = 'L' or 'l', it must be at
 *        least max(1,m) and at least max(1,n) otherwise
 * B      single precision array of dimensions (ldb, n). On entry, the 
 *        leading m x n part of the array contains the matrix B. It is
 *        overwritten with the transformed matrix on exit.
 * ldb    leading dimension of B. It must be at least max (1, m).
 *
 * Output
 * ------
 * B      updated according to B = alpha * op(A) * B  or B = alpha * B * op(A)
 *
 * Reference: http://www.netlib.org/blas/strmm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasStrmm (char side, char uplo, char transa,
                                     char diag, int m, int n, float alpha,
                                     const float *A, int lda, float *B,
                                     int ldb)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasStrmmParams params;
    cudaError_t cudaStat;
    int info;
    int lside = toupper(side) == 'L';
    int upper, notrans, unit, nrowa;
    int funcIdx;
    int useFastImul;
    int fullTilesOnly;
    int usePureHwStepper;
    dim3 ctaDimsHw (lside ? ((n+BLK-1)/BLK) : ((m+BLK-1)/BLK));
    dim3 ctaDimsSw (CUBLAS_STRMM_CTAS);

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }

    upper   = toupper(uplo)   == 'U';
    notrans = toupper(transa) == 'N';
    unit    = toupper(diag)   == 'U';

    nrowa = (lside) ? m : n;

    info = 0;
    if ((!lside) && (toupper(side) != 'R')) {
        info = 1;
    } 
    else if ((!upper) && (toupper(uplo) != 'L')) {
        info = 2;
    }
    else if ((!notrans) && (toupper(transa) != 'T') && (toupper(transa)!='C')){
        info = 3;
    }
    else if ((unit) && (toupper(diag) != 'U')) {
        info = 4;
    }
    else if (m < 0) {
        info = 5;
    }
    else if (n < 0) {
        info = 6;
    }
    else if (lda < imax (1, nrowa)) {
        info = 9;
    }
    else if (ldb < imax (1, m)) {
        info = 11;
    }
    if (info) {
        cublasXerbla ("STRMM ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((m == 0) || (n == 0)) return;

    params.lside = lside;
    params.upper = upper;
    params.notrans = notrans;
    params.unit = unit;
    params.m = m;
    params.n = n;
    params.alpha = alpha;
    params.A = A;
    params.lda = lda;
    params.B = B;
    params.ldb = ldb;

    /* choose HW-only stepping if columns in result matrix do not exceed the
     * maximum CTA grid dimensions.
     */
    usePureHwStepper = ((m < (CUBLAS_CTA_MAX_DIM * BLK)) &&
                        (n < (CUBLAS_CTA_MAX_DIM * BLK)));

    /* We can eliminate checking for endcases if we know all tiles are fully
     * populated. Important benchmark case!
     */
    fullTilesOnly = (((m % BLK) == 0) && ((n % BLK) == 0));

    /* choose version using 24-bit multiplies if all dimensions are less than
     * 2001, so we can guarantee that no multiplication result exceeds (2000 *
     * 2000 * 4) < 2^24.
     */
    useFastImul = ((params.lda <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                   (params.ldb <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                   (params.m <= CUBLAS_FASTIMUL_F_MAX_DIM) && 
                   (params.n <= CUBLAS_FASTIMUL_F_MAX_DIM));

    funcIdx = ((fullTilesOnly << 6) | (useFastImul << 5) | 
               ((params.alpha == 0.0f) << 4) | 
               (params.unit << 3) | (params.lside << 2) | 
               (params.notrans << 1) | params.upper);

    cudaStat = cudaGetLastError(); /* clear error status */
    if (usePureHwStepper) {
        strmm_hw[funcIdx]<<<ctaDimsHw,CUBLAS_STRMM_THREAD_COUNT>>>(params);
    } else {
        strmm_sw[funcIdx]<<<ctaDimsSw,CUBLAS_STRMM_THREAD_COUNT>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}


__shared__ float AA[(BLK+1)*BLK];  // padded to avoid GRF bank conflicts
__shared__ float BB[(BLK+1)*BLK];  // padded to avoid GRF bank conflicts

__global__ void strmm_l_up_nt_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALHPA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_l_up_nt_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_l_up_nt_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0 
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_l_up_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALHPA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_unit_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1 
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_unit_alpha0_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}


__global__ void strmm_l_up_nt_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALHPA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_l_up_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_l_up_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0 
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_l_up_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALHPA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_unit_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1 
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_unit_alpha0_fulltile_sw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 1
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_l_up_nt_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALHPA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_l_up_nt_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_l_up_nt_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0 
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_l_up_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALHPA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_unit_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1 
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_unit_alpha0_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   0
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}


__global__ void strmm_l_up_nt_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALHPA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_l_up_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_l_up_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0 
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_l_up_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_lo_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void strmm_l_up_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_l_lo_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void strmm_r_up_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_lo_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void strmm_r_up_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void strmm_r_lo_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         0
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALHPA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_unit_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            0
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1 
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              0
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_l_up_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_l.h"
}

__global__ void fast_strmm_l_up_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_l_lo_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_l.h"
}

__global__ void fast_strmm_r_up_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_nt_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             0
#include "strmm_r.h"
}

__global__ void fast_strmm_r_up_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             0
#define TRANS             1
#include "strmm_r.h"
}

__global__ void fast_strmm_r_lo_tr_main_unit_alpha0_fulltile_hw (struct cublasStrmmParams parms) 
{
#undef  USE_MIXED_STEPPER
#undef  FULL_TILES_ONLY
#undef  FAST_IMUL
#undef  ALPHA0
#undef  UNIT
#undef  LOWER
#undef  TRANS
#define USE_MIXED_STEPPER 0
#define FULL_TILES_ONLY   1
#define FAST_IMUL         1
#define ALPHA0            1
#define UNIT              1
#define LOWER             1
#define TRANS             1
#include "strmm_r.h"
}
