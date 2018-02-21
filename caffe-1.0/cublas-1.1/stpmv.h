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

    /* NOTE: wrapper must ensure that parms.n >= 0, and parms.incx != 0 */
    int tid;
    int startx;
    int i = 0;
    
    tid = threadIdx.x;
    startx = (parms.incx >= 0) ? 0 : ((1 - parms.n) * parms.incx);
    if (blockIdx.x == 0) {

#if (UP==0)
        int twoN = parms.n << 1;
#undef  IDXA
#define IDXA(row,col)    ((row) + (((twoN - (col) - 1) * (col)) >> 1))
#else
#undef  IDXA
#define IDXA(row,col)    ((row) + (((col) * (col) + (col)) >> 1))
#endif
        int addr = IDXX(tid+0*XINC);
        int incr = XINC * parms.incx;
        __syncthreads();
        /* copy x into GRF */
        for (i = tid; i < parms.n; i += XINC) {
            xx[i] = parms.x[addr];
            addr += incr;
        }
        __syncthreads();

        for (i = tid; i < parms.n; i += XINC) {
            float dp = 0.0f;
            int idx;
            int j;
            for (j = 0; j < parms.n; j++) {
                float Aij = (i == j) ? 1.0f : 0.0f;
#if ((UP==1)^(TRANS==1))
                if ((i < j) || ((i == j) && (!parms.unit))) {
#else
                if ((i > j) || ((i == j) && (!parms.unit))) {    
#endif
#if (UP==1)
                    idx = (i <= j) ? IDXA (i, j) : IDXA (j, i);
#else
                    idx = (i >= j) ? IDXA (i, j) : IDXA (j, i);
#endif
                    Aij = parms.AP[idx];
                }
                dp += Aij * xx[j];
            }
            parms.x[IDXX(i)] = dp;
        }
    }
