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

    /*
     * NOTE: wrapper must ensure that parms.n >= 0, and parms.incx != 0 
     */
    int tid;
    int startx;
    int i = 0;
    
    tid = threadIdx.x;
    startx = (parms.incx >= 0) ? 0 : ((1 - parms.n) * parms.incx);
    if (blockIdx.x == 0) {

#if (UP==1)
#undef  IDXA
#define IDXA(row,col)    ((parms.lda*(col))+(parms.k)+((row)-(col)))
#else
#undef  IDXA
#define IDXA(row,col)    ((parms.lda*(col))+((row)-(col)))
#endif
        int addr = IDXX(tid+0*XINC);
        int idx  = tid+0*XINC;
        int incr = XINC * parms.incx;
        __syncthreads();
        /* copy x into GRF */
        for (i = tid; i < parms.n; i += XINC) {
            xx[idx] = parms.x[addr];
            addr += incr;
            idx  += XINC;
        }
        __syncthreads();
        for (i = tid; i < parms.n; i += XINC) {
            float dp = xx[i];
            int j;
            int start, limit;
            if (!parms.unit) {
                dp *= parms.A[IDXA(i,i)];
            }
#if (UP^TRANS)
            start = i + 1;
            limit = min(parms.n, (i+parms.k+1));
#else
            start = max(0,i-parms.k);
            limit = i;
#endif
            for (j = start; j < limit; j++) {
#if (TRANS)
                dp += parms.A[IDXA(j,i)] * xx[j];
#else                    
                dp += parms.A[IDXA(i,j)] * xx[j];
#endif
            }
            parms.x[IDXX(i)] = dp;
        }
    }



