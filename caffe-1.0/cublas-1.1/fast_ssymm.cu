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

/* This file contains the implementation of the BLAS-3 function ssymm */

#include "ssymm_common.h" /* shared between ssymm.cu and fast_ssymm.cu */

__global__ void fast_ssymm_main_hw_lo_right (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             0
#define LSIDE             0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void fast_ssymm_main_hw_lo_left (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             0
#define LSIDE             1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void fast_ssymm_main_hw_up_right (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             1
#define LSIDE             0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void fast_ssymm_main_hw_up_left (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             1
#define LSIDE             1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void fast_ssymm_main_sw_lo_right (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             0
#define LSIDE             0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void fast_ssymm_main_sw_lo_left (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             0
#define LSIDE             1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void fast_ssymm_main_sw_up_right (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             1
#define LSIDE             0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void fast_ssymm_main_sw_up_left (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             1
#define LSIDE             1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   0
#include "ssymm.h"
}

__global__ void fast_ssymm_main_hw_lo_right_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             0
#define LSIDE             0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void fast_ssymm_main_hw_lo_left_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             0
#define LSIDE             1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void fast_ssymm_main_hw_up_right_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             1
#define LSIDE             0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void fast_ssymm_main_hw_up_left_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 0
#define UPPER             1
#define LSIDE             1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void fast_ssymm_main_sw_lo_right_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             0
#define LSIDE             0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void fast_ssymm_main_sw_lo_left_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             0
#define LSIDE             1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void fast_ssymm_main_sw_up_right_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             1
#define LSIDE             0
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}

__global__ void fast_ssymm_main_sw_up_left_fulltile (struct cublasSsymmParams parms)
{
#undef  USE_MIXED_STEPPER
#undef  UPPER
#undef  LSIDE
#undef  FAST_IMUL
#undef  FULL_TILES_ONLY
#define USE_MIXED_STEPPER 1
#define UPPER             1
#define LSIDE             1
#define FAST_IMUL         1
#define FULL_TILES_ONLY   1
#include "ssymm.h"
}
