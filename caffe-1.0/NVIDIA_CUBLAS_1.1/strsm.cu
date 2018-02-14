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

/* This file contains the implementation of the BLAS-3 function strsm */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

__global__ void strsm_l_up_tr_main_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);

__global__ void strsm_l_up_tr_main_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_fulltile_sw_nu(struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_fulltile_sw_nu(struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_fulltile_sw_nu(struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_up_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_l_lo_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_alpha0_sw_nu(struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_up_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_l_lo_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);



__global__ void strsm_r_up_tr_main_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_fulltile_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_fulltile_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms);

__global__ void strsm_r_up_tr_main_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_up_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void strsm_r_lo_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_up_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);
__global__ void fast_strsm_r_lo_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms);


typedef void (*pf) (struct cublasStrsmParams parms);

static pf strsm_l_hw[64] = {
    strsm_l_lo_tr_main_hw,
    strsm_l_lo_nt_main_hw,
    strsm_l_up_tr_main_hw,
    strsm_l_up_nt_main_hw,
    strsm_l_lo_tr_main_fulltile_hw,
    strsm_l_lo_nt_main_fulltile_hw,
    strsm_l_up_tr_main_fulltile_hw,
    strsm_l_up_nt_main_fulltile_hw,
    strsm_l_lo_tr_main_alpha0_hw,
    strsm_l_lo_nt_main_alpha0_hw,
    strsm_l_up_tr_main_alpha0_hw,
    strsm_l_up_nt_main_alpha0_hw,
    strsm_l_lo_tr_main_fulltile_alpha0_hw,
    strsm_l_lo_nt_main_fulltile_alpha0_hw,
    strsm_l_up_tr_main_fulltile_alpha0_hw,
    strsm_l_up_nt_main_fulltile_alpha0_hw,
    fast_strsm_l_lo_tr_main_hw,
    fast_strsm_l_lo_nt_main_hw,
    fast_strsm_l_up_tr_main_hw,
    fast_strsm_l_up_nt_main_hw,
    fast_strsm_l_lo_tr_main_fulltile_hw,
    fast_strsm_l_lo_nt_main_fulltile_hw,
    fast_strsm_l_up_tr_main_fulltile_hw,
    fast_strsm_l_up_nt_main_fulltile_hw,
    fast_strsm_l_lo_tr_main_alpha0_hw,
    fast_strsm_l_lo_nt_main_alpha0_hw,
    fast_strsm_l_up_tr_main_alpha0_hw,
    fast_strsm_l_up_nt_main_alpha0_hw,
    fast_strsm_l_lo_tr_main_fulltile_alpha0_hw,
    fast_strsm_l_lo_nt_main_fulltile_alpha0_hw,
    fast_strsm_l_up_tr_main_fulltile_alpha0_hw,
    fast_strsm_l_up_nt_main_fulltile_alpha0_hw,
    strsm_l_lo_tr_main_hw_nu,
    strsm_l_lo_nt_main_hw_nu,
    strsm_l_up_tr_main_hw_nu,
    strsm_l_up_nt_main_hw_nu,
    strsm_l_lo_tr_main_fulltile_hw_nu,
    strsm_l_lo_nt_main_fulltile_hw_nu,
    strsm_l_up_tr_main_fulltile_hw_nu,
    strsm_l_up_nt_main_fulltile_hw_nu,
    strsm_l_lo_tr_main_alpha0_hw_nu,
    strsm_l_lo_nt_main_alpha0_hw_nu,
    strsm_l_up_tr_main_alpha0_hw_nu,
    strsm_l_up_nt_main_alpha0_hw_nu,
    strsm_l_lo_tr_main_fulltile_alpha0_hw_nu,
    strsm_l_lo_nt_main_fulltile_alpha0_hw_nu,
    strsm_l_up_tr_main_fulltile_alpha0_hw_nu,
    strsm_l_up_nt_main_fulltile_alpha0_hw_nu,
    fast_strsm_l_lo_tr_main_hw_nu,
    fast_strsm_l_lo_nt_main_hw_nu,
    fast_strsm_l_up_tr_main_hw_nu,
    fast_strsm_l_up_nt_main_hw_nu,
    fast_strsm_l_lo_tr_main_fulltile_hw_nu,
    fast_strsm_l_lo_nt_main_fulltile_hw_nu,
    fast_strsm_l_up_tr_main_fulltile_hw_nu,
    fast_strsm_l_up_nt_main_fulltile_hw_nu,
    fast_strsm_l_lo_tr_main_alpha0_hw_nu,
    fast_strsm_l_lo_nt_main_alpha0_hw_nu,
    fast_strsm_l_up_tr_main_alpha0_hw_nu,
    fast_strsm_l_up_nt_main_alpha0_hw_nu,
    fast_strsm_l_lo_tr_main_fulltile_alpha0_hw_nu,
    fast_strsm_l_lo_nt_main_fulltile_alpha0_hw_nu,
    fast_strsm_l_up_tr_main_fulltile_alpha0_hw_nu,
    fast_strsm_l_up_nt_main_fulltile_alpha0_hw_nu,
};

static pf strsm_l_sw[64] = {
    strsm_l_lo_tr_main_sw,
    strsm_l_lo_nt_main_sw,
    strsm_l_up_tr_main_sw,
    strsm_l_up_nt_main_sw,
    strsm_l_lo_tr_main_fulltile_sw,
    strsm_l_lo_nt_main_fulltile_sw,
    strsm_l_up_tr_main_fulltile_sw,
    strsm_l_up_nt_main_fulltile_sw,
    strsm_l_lo_tr_main_alpha0_sw,
    strsm_l_lo_nt_main_alpha0_sw,
    strsm_l_up_tr_main_alpha0_sw,
    strsm_l_up_nt_main_alpha0_sw,
    strsm_l_lo_tr_main_fulltile_alpha0_sw,
    strsm_l_lo_nt_main_fulltile_alpha0_sw,
    strsm_l_up_tr_main_fulltile_alpha0_sw,
    strsm_l_up_nt_main_fulltile_alpha0_sw,
    fast_strsm_l_lo_tr_main_sw,
    fast_strsm_l_lo_nt_main_sw,
    fast_strsm_l_up_tr_main_sw,
    fast_strsm_l_up_nt_main_sw,
    fast_strsm_l_lo_tr_main_fulltile_sw,
    fast_strsm_l_lo_nt_main_fulltile_sw,
    fast_strsm_l_up_tr_main_fulltile_sw,
    fast_strsm_l_up_nt_main_fulltile_sw,
    fast_strsm_l_lo_tr_main_alpha0_sw,
    fast_strsm_l_lo_nt_main_alpha0_sw,
    fast_strsm_l_up_tr_main_alpha0_sw,
    fast_strsm_l_up_nt_main_alpha0_sw,
    fast_strsm_l_lo_tr_main_fulltile_alpha0_sw,
    fast_strsm_l_lo_nt_main_fulltile_alpha0_sw,
    fast_strsm_l_up_tr_main_fulltile_alpha0_sw,
    fast_strsm_l_up_nt_main_fulltile_alpha0_sw,
    strsm_l_lo_tr_main_sw_nu,
    strsm_l_lo_nt_main_sw_nu,
    strsm_l_up_tr_main_sw_nu,
    strsm_l_up_nt_main_sw_nu,
    strsm_l_lo_tr_main_fulltile_sw_nu,
    strsm_l_lo_nt_main_fulltile_sw_nu,
    strsm_l_up_tr_main_fulltile_sw_nu,
    strsm_l_up_nt_main_fulltile_sw_nu,
    strsm_l_lo_tr_main_alpha0_sw_nu,
    strsm_l_lo_nt_main_alpha0_sw_nu,
    strsm_l_up_tr_main_alpha0_sw_nu,
    strsm_l_up_nt_main_alpha0_sw_nu,
    strsm_l_lo_tr_main_fulltile_alpha0_sw_nu,
    strsm_l_lo_nt_main_fulltile_alpha0_sw_nu,
    strsm_l_up_tr_main_fulltile_alpha0_sw_nu,
    strsm_l_up_nt_main_fulltile_alpha0_sw_nu,
    fast_strsm_l_lo_tr_main_sw_nu,
    fast_strsm_l_lo_nt_main_sw_nu,
    fast_strsm_l_up_tr_main_sw_nu,
    fast_strsm_l_up_nt_main_sw_nu,
    fast_strsm_l_lo_tr_main_fulltile_sw_nu,
    fast_strsm_l_lo_nt_main_fulltile_sw_nu,
    fast_strsm_l_up_tr_main_fulltile_sw_nu,
    fast_strsm_l_up_nt_main_fulltile_sw_nu,
    fast_strsm_l_lo_tr_main_alpha0_sw_nu,
    fast_strsm_l_lo_nt_main_alpha0_sw_nu,
    fast_strsm_l_up_tr_main_alpha0_sw_nu,
    fast_strsm_l_up_nt_main_alpha0_sw_nu,
    fast_strsm_l_lo_tr_main_fulltile_alpha0_sw_nu,
    fast_strsm_l_lo_nt_main_fulltile_alpha0_sw_nu,
    fast_strsm_l_up_tr_main_fulltile_alpha0_sw_nu,
    fast_strsm_l_up_nt_main_fulltile_alpha0_sw_nu,
};

static pf strsm_r_hw[64] = {
    strsm_r_lo_tr_main_hw,
    strsm_r_lo_nt_main_hw,
    strsm_r_up_tr_main_hw,
    strsm_r_up_nt_main_hw,
    strsm_r_lo_tr_main_fulltile_hw,
    strsm_r_lo_nt_main_fulltile_hw,
    strsm_r_up_tr_main_fulltile_hw,
    strsm_r_up_nt_main_fulltile_hw,
    strsm_r_lo_tr_main_alpha0_hw,
    strsm_r_lo_nt_main_alpha0_hw,
    strsm_r_up_tr_main_alpha0_hw,
    strsm_r_up_nt_main_alpha0_hw,
    strsm_r_lo_tr_main_fulltile_alpha0_hw,
    strsm_r_lo_nt_main_fulltile_alpha0_hw,
    strsm_r_up_tr_main_fulltile_alpha0_hw,
    strsm_r_up_nt_main_fulltile_alpha0_hw,
    fast_strsm_r_lo_tr_main_hw,
    fast_strsm_r_lo_nt_main_hw,
    fast_strsm_r_up_tr_main_hw,
    fast_strsm_r_up_nt_main_hw,
    fast_strsm_r_lo_tr_main_fulltile_hw,
    fast_strsm_r_lo_nt_main_fulltile_hw,
    fast_strsm_r_up_tr_main_fulltile_hw,
    fast_strsm_r_up_nt_main_fulltile_hw,
    fast_strsm_r_lo_tr_main_alpha0_hw,
    fast_strsm_r_lo_nt_main_alpha0_hw,
    fast_strsm_r_up_tr_main_alpha0_hw,
    fast_strsm_r_up_nt_main_alpha0_hw,
    fast_strsm_r_lo_tr_main_fulltile_alpha0_hw,
    fast_strsm_r_lo_nt_main_fulltile_alpha0_hw,
    fast_strsm_r_up_tr_main_fulltile_alpha0_hw,
    fast_strsm_r_up_nt_main_fulltile_alpha0_hw,
    strsm_r_lo_tr_main_hw_nu,
    strsm_r_lo_nt_main_hw_nu,
    strsm_r_up_tr_main_hw_nu,
    strsm_r_up_nt_main_hw_nu,
    strsm_r_lo_tr_main_fulltile_hw_nu,
    strsm_r_lo_nt_main_fulltile_hw_nu,
    strsm_r_up_tr_main_fulltile_hw_nu,
    strsm_r_up_nt_main_fulltile_hw_nu,
    strsm_r_lo_tr_main_alpha0_hw_nu,
    strsm_r_lo_nt_main_alpha0_hw_nu,
    strsm_r_up_tr_main_alpha0_hw_nu,
    strsm_r_up_nt_main_alpha0_hw_nu,
    strsm_r_lo_tr_main_fulltile_alpha0_hw_nu,
    strsm_r_lo_nt_main_fulltile_alpha0_hw_nu,
    strsm_r_up_tr_main_fulltile_alpha0_hw_nu,
    strsm_r_up_nt_main_fulltile_alpha0_hw_nu,
    fast_strsm_r_lo_tr_main_hw_nu,
    fast_strsm_r_lo_nt_main_hw_nu,
    fast_strsm_r_up_tr_main_hw_nu,
    fast_strsm_r_up_nt_main_hw_nu,
    fast_strsm_r_lo_tr_main_fulltile_hw_nu,
    fast_strsm_r_lo_nt_main_fulltile_hw_nu,
    fast_strsm_r_up_tr_main_fulltile_hw_nu,
    fast_strsm_r_up_nt_main_fulltile_hw_nu,
    fast_strsm_r_lo_tr_main_alpha0_hw_nu,
    fast_strsm_r_lo_nt_main_alpha0_hw_nu,
    fast_strsm_r_up_tr_main_alpha0_hw_nu,
    fast_strsm_r_up_nt_main_alpha0_hw_nu,
    fast_strsm_r_lo_tr_main_fulltile_alpha0_hw_nu,
    fast_strsm_r_lo_nt_main_fulltile_alpha0_hw_nu,
    fast_strsm_r_up_tr_main_fulltile_alpha0_hw_nu,
    fast_strsm_r_up_nt_main_fulltile_alpha0_hw_nu
};

static pf strsm_r_sw[64] = {
    strsm_r_lo_tr_main_sw,
    strsm_r_lo_nt_main_sw,
    strsm_r_up_tr_main_sw,
    strsm_r_up_nt_main_sw,
    strsm_r_lo_tr_main_fulltile_sw,
    strsm_r_lo_nt_main_fulltile_sw,
    strsm_r_up_tr_main_fulltile_sw,
    strsm_r_up_nt_main_fulltile_sw,
    strsm_r_lo_tr_main_alpha0_sw,
    strsm_r_lo_nt_main_alpha0_sw,
    strsm_r_up_tr_main_alpha0_sw,
    strsm_r_up_nt_main_alpha0_sw,
    strsm_r_lo_tr_main_fulltile_alpha0_sw,
    strsm_r_lo_nt_main_fulltile_alpha0_sw,
    strsm_r_up_tr_main_fulltile_alpha0_sw,
    strsm_r_up_nt_main_fulltile_alpha0_sw,
    fast_strsm_r_lo_tr_main_sw,
    fast_strsm_r_lo_nt_main_sw,
    fast_strsm_r_up_tr_main_sw,
    fast_strsm_r_up_nt_main_sw,
    fast_strsm_r_lo_tr_main_fulltile_sw,
    fast_strsm_r_lo_nt_main_fulltile_sw,
    fast_strsm_r_up_tr_main_fulltile_sw,
    fast_strsm_r_up_nt_main_fulltile_sw,
    fast_strsm_r_lo_tr_main_alpha0_sw,
    fast_strsm_r_lo_nt_main_alpha0_sw,
    fast_strsm_r_up_tr_main_alpha0_sw,
    fast_strsm_r_up_nt_main_alpha0_sw,
    fast_strsm_r_lo_tr_main_fulltile_alpha0_sw,
    fast_strsm_r_lo_nt_main_fulltile_alpha0_sw,
    fast_strsm_r_up_tr_main_fulltile_alpha0_sw,
    fast_strsm_r_up_nt_main_fulltile_alpha0_sw,
    strsm_r_lo_tr_main_sw_nu,
    strsm_r_lo_nt_main_sw_nu,
    strsm_r_up_tr_main_sw_nu,
    strsm_r_up_nt_main_sw_nu,
    strsm_r_lo_tr_main_fulltile_sw_nu,
    strsm_r_lo_nt_main_fulltile_sw_nu,
    strsm_r_up_tr_main_fulltile_sw_nu,
    strsm_r_up_nt_main_fulltile_sw_nu,
    strsm_r_lo_tr_main_alpha0_sw_nu,
    strsm_r_lo_nt_main_alpha0_sw_nu,
    strsm_r_up_tr_main_alpha0_sw_nu,
    strsm_r_up_nt_main_alpha0_sw_nu,
    strsm_r_lo_tr_main_fulltile_alpha0_sw_nu,
    strsm_r_lo_nt_main_fulltile_alpha0_sw_nu,
    strsm_r_up_tr_main_fulltile_alpha0_sw_nu,
    strsm_r_up_nt_main_fulltile_alpha0_sw_nu,
    fast_strsm_r_lo_tr_main_sw_nu,
    fast_strsm_r_lo_nt_main_sw_nu,
    fast_strsm_r_up_tr_main_sw_nu,
    fast_strsm_r_up_nt_main_sw_nu,
    fast_strsm_r_lo_tr_main_fulltile_sw_nu,
    fast_strsm_r_lo_nt_main_fulltile_sw_nu,
    fast_strsm_r_up_tr_main_fulltile_sw_nu,
    fast_strsm_r_up_nt_main_fulltile_sw_nu,
    fast_strsm_r_lo_tr_main_alpha0_sw_nu,
    fast_strsm_r_lo_nt_main_alpha0_sw_nu,
    fast_strsm_r_up_tr_main_alpha0_sw_nu,
    fast_strsm_r_up_nt_main_alpha0_sw_nu,
    fast_strsm_r_lo_tr_main_fulltile_alpha0_sw_nu,
    fast_strsm_r_lo_nt_main_fulltile_alpha0_sw_nu,
    fast_strsm_r_up_tr_main_fulltile_alpha0_sw_nu,
    fast_strsm_r_up_nt_main_fulltile_alpha0_sw_nu
};

#define BLK_LOG             (5)
#define BLK                 (1 << BLK_LOG)
#define JINC                (BLK * CUBLAS_STRSM_CTAS)
#define IINC                (BLK * CUBLAS_STRSM_CTAS)

/*
 * void 
 * cublasStrsm (char side, char uplo, char transa, char diag, int m, int n, 
 *              float alpha, const float *A, int lda, float *B, int ldb)
 *
 * solves one of the matrix equations
 *
 *    op(A) * X = alpha * B,   or   X * op(A) = alpha * B,
 *
 * where alpha is a single precision scalar, and X and B are m x n matrices 
 * that are composed of single precision elements. A is a unit or non-unit,
 * upper or lower triangular matrix, and op(A) is one of 
 *
 *    op(A) = A  or  op(A) = transpose(A)
 *
 * The result matrix X overwrites input matrix B; that is, on exit the result 
 * is stored in B. Matrices A and B are stored in column major format, and
 * lda and ldb are the leading dimensions of the two-dimensonials arrays that
 * contain A and B, respectively.
 *
 * Input
 * -----
 * side   specifies whether op(A) appears on the left or right of X as
 *        follows: side = 'L' or 'l' indicates solve op(A) * X = alpha * B.
 *        side = 'R' or 'r' indicates solve X * op(A) = alpha * B.
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix as follows: uplo = 'U' or 'u' indicates A is an upper
 *        triangular matrix. uplo = 'L' or 'l' indicates A is a lower
 *        triangular matrix.
 * transa specifies the form of op(A) to be used in matrix multiplication
 *        as follows: If transa = 'N' or 'N', then op(A) = A. If transa =
 *        'T', 't', 'C', or 'c', then op(A) = transpose(A).
 * diag   specifies whether or not A is a unit triangular matrix like so:
 *        if diag = 'U' or 'u', A is assumed to be unit triangular. If 
 *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
 * m      specifies the number of rows of B. m must be at least zero.
 * n      specifies the number of columns of B. n must be at least zero.
 * alpha  is a single precision scalar to be multiplied with B. When alpha is 
 *        zero, then A is not referenced and B need not be set before entry.
 * A      is a single precision array of dimensions (lda, k), where k is
 *        m when side = 'L' or 'l', and is n when side = 'R' or 'r'. If
 *        uplo = 'U' or 'u', the leading k x k upper triangular part of
 *        the array A must contain the upper triangular matrix and the
 *        strictly lower triangular matrix of A is not referenced. When
 *        uplo = 'L' or 'l', the leading k x k lower triangular part of
 *        the array A must contain the lower triangular matrix and the 
 *        strictly upper triangular part of A is not referenced. Note that
 *        when diag = 'U' or 'u', the diagonal elements of A are not
 *        referenced, and are assumed to be unity.
 * lda    is the leading dimension of the two dimensional array containing A.
 *        When side = 'L' or 'l' then lda must be at least max(1, m), when 
 *        side = 'R' or 'r' then lda must be at least max(1, n).
 * B      is a single precision array of dimensions (ldb, n). ldb must be
 *        at least max (1,m). The leading m x n part of the array B must 
 *        contain the right-hand side matrix B. On exit B is overwritten 
 *        by the solution matrix X.
 * ldb    is the leading dimension of the two dimensional array containing B.
 *        ldb must be at least max(1, m).
 *
 * Output
 * ------
 * B      contains the solution matrix X satisfying op(A) * X = alpha * B, 
 *        or X * op(A) = alpha * B
 *
 * Reference: http://www.netlib.org/blas/strsm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasStrsm (char side, char uplo, char transa, 
                                     char diag, int m, int n, float alpha,
                                     const float *A, int lda, float *B,
                                     int ldb)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasStrsmParams params;
    cudaError_t cudaStat;
    int fullTilesOnly;
    int funcIdx;
    int useFastImul;
    int usePureHwStepper;
    int info;
    int lside = toupper(side)   == 'L';
    int upper, notrans, nounit, nrowa;
    dim3 ctaDimsHw (lside ? ((n+BLK-1)/BLK) : ((m+BLK-1)/BLK));
    dim3 ctaDimsSw (CUBLAS_STRSM_CTAS);

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }

    upper   = toupper(uplo)   == 'U';
    notrans = toupper(transa) == 'N';
    nounit  = toupper(diag)   == 'N';

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
    else if ((!nounit) && (toupper(diag) != 'U')) {
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
        cublasXerbla ("STRSM ", info);
        cublasSetError (ctx, CUBLAS_STATUS_INVALID_VALUE);
        return;
    }

    /* early out if nothing to do */
    if ((m == 0) || (n == 0)) {
        return;
    }

    params.lside = lside;
    params.upper = upper;
    params.notrans = notrans;
    params.nounit = nounit;
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
    usePureHwStepper = (params.lside ? (n < (CUBLAS_CTA_MAX_DIM * BLK)) :
                                       (m < (CUBLAS_CTA_MAX_DIM * BLK)));

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

    funcIdx = ((params.nounit << 5) | (useFastImul << 4) | 
               ((params.alpha == 0.0f) << 3) | (fullTilesOnly << 2) | 
               (params.upper << 1) | params.notrans);

    cudaStat = cudaGetLastError(); /* clear error status */
    if (params.lside) {
        if (usePureHwStepper) {
            strsm_l_hw[funcIdx]<<<ctaDimsHw,CUBLAS_STRSM_THREAD_COUNT>>>(params);
        } else {
            strsm_l_sw[funcIdx]<<<ctaDimsSw,CUBLAS_STRSM_THREAD_COUNT>>>(params);
        }
    } else {
        if (usePureHwStepper) {
            strsm_r_hw[funcIdx]<<<ctaDimsHw,CUBLAS_STRSM_THREAD_COUNT>>>(params);
        } else {
            strsm_r_sw[funcIdx]<<<ctaDimsSw,CUBLAS_STRSM_THREAD_COUNT>>>(params);
        }
    }
    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }
}


#if ((CUBLAS_STRSM_THREAD_COUNT<BLK))
#error block dimension must be >= threadcount
#endif

#if ((CUBLAS_STRSM_THREAD_COUNT%BLK)!=0)
#error threadcount and block dimensions do not divide evenly
#endif

#define A_NBR_COLS          (CUBLAS_STRSM_THREAD_COUNT/BLK)
#define B_NBR_COLS          (CUBLAS_STRSM_THREAD_COUNT/BLK)

#if (((BLK*BLK)%CUBLAS_STRSM_THREAD_COUNT)!=0)
#error blocksize of A and B not evenly divided by threadcount!
#endif

#define A_ELEMS_PER_THREAD  ((BLK * BLK) / CUBLAS_STRSM_THREAD_COUNT)
#define B_ELEMS_PER_THREAD  ((BLK * BLK) / CUBLAS_STRSM_THREAD_COUNT)

__shared__ float AA[(BLK+1)*BLK];  // padded to avoid GRF bank conflicts
__shared__ float BB[(BLK+1)*BLK];  // padded to avoid GRF bank conflicts

__global__ void strsm_l_lo_nt_main_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_lo_nt_main_fulltile_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_fulltile_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_fulltile_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_fulltile_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_lo_nt_main_alpha0_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}
__global__ void strsm_l_lo_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_fulltile_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_fulltile_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_fulltile_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_fulltile_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_alpha0_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_lo_nt_main_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}
__global__ void strsm_l_lo_nt_main_fulltile_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_fulltile_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_fulltile_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_fulltile_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_lo_nt_main_alpha0_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}
__global__ void strsm_l_lo_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_fulltile_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_fulltile_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_fulltile_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_fulltile_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_alpha0_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

/*-----------------------------------------------------*/
__global__ void strsm_l_lo_nt_main_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_lo_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_lo_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}
__global__ void strsm_l_lo_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_lo_nt_main_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}
__global__ void strsm_l_lo_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_lo_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}
__global__ void strsm_l_lo_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_lo_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_l_up_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void strsm_l_up_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY 
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_lo_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_l.h"
}

__global__ void fast_strsm_l_up_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_l.h"
}

__global__ void strsm_r_lo_nt_main_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_lo_tr_main_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_fulltile_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_lo_tr_main_fulltile_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_fulltile_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_fulltile_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_alpha0_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_lo_tr_main_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_lo_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}
__global__ void fast_strsm_r_lo_tr_main_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_fulltile_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_tr_main_fulltile_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_fulltile_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_fulltile_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_alpha0_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_tr_main_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_fulltile_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_fulltile_alpha0_sw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}
__global__ void strsm_r_lo_tr_main_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_fulltile_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_lo_tr_main_fulltile_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_fulltile_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_fulltile_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_alpha0_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_lo_tr_main_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_lo_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}
__global__ void fast_strsm_r_lo_tr_main_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_fulltile_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_tr_main_fulltile_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_fulltile_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_fulltile_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_alpha0_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_tr_main_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_fulltile_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_fulltile_alpha0_hw (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            0
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}
__global__ void strsm_r_lo_tr_main_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_lo_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_lo_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_lo_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}
__global__ void fast_strsm_r_lo_tr_main_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_fulltile_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_fulltile_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_fulltile_alpha0_sw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 1
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}
 
__global__ void strsm_r_lo_nt_main_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}
__global__ void strsm_r_lo_tr_main_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_lo_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_lo_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_lo_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_lo_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void strsm_r_up_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void strsm_r_up_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         0
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}
__global__ void fast_strsm_r_lo_tr_main_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_fulltile_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_fulltile_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     0
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   0
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms) 
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_lo_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             1
#define TRANS             1
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_nt_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             0
#include "strsm_r.h"
}

__global__ void fast_strsm_r_up_tr_main_fulltile_alpha0_hw_nu (struct cublasStrsmParams parms)
{
#undef NOUNIT
#undef USE_MIXED_STEPPER
#undef FAST_IMUL
#undef ALPHA_IS_ZERO
#undef FULL_TILES_ONLY
#undef LOWER
#undef TRANS
#define NOUNIT            1
#define USE_MIXED_STEPPER 0
#define FAST_IMUL         1
#define ALPHA_IS_ZERO     1
#define FULL_TILES_ONLY   1
#define LOWER             0
#define TRANS             1
#include "strsm_r.h"
}
