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

#if (UP==0)
#undef  IDXA
#define IDXA(row,col)  ((parms.lda*(col))+((row)-(col)))
#else
#undef  IDXA
#define IDXA(row,col)  ((parms.lda*(col))+(parms.k)+((row)-(col)))
#endif
    /*
     * NOTE: wrapper must ensure that parms.n >= 0, and parms.incx != 0 
     */
    int tid;
    int startx;
    int i;
    
    tid = threadIdx.x;
    startx = (parms.incx >= 0) ? 0 : ((1 - parms.n) * parms.incx);
    
    if (blockIdx.x == 0) {
        int addr = IDXX(tid);
        int incr = XINC * parms.incx;
        __syncthreads();
        /* copy x from GMEM into GRF */
        for (i = tid; i < parms.n; i += XINC) {
            XX[i] = parms.x[addr];
            addr += incr;
        }
        __syncthreads();
        
#if ((UP==1)^(TRANS==1))
        for (i = (parms.n - 1); i >= 0; i--) {
#else
        for (i = 0; i < parms.n; i++) {
#endif
            __syncthreads();
            if (tid == 0) {
                temp = XX[i];
                if (!parms.unit) {
                    temp /= parms.A[IDXA(i,i)];
                }
                XX[i] = temp;
            }
            __syncthreads();

#if ((UP==1)^(TRANS==1))
            int start = tid;
            int limit = i;
#else
            int start = (i+1)+tid;
            int limit = parms.n;
#endif
            int x;
            for (x = start; x < limit; x += XINC) {
#if ((UP==1)^(TRANS==1))
                if (x >= (i - parms.k)) {
#else
                if (x <= (i + parms.k)) {
#endif
#if (TRANS==0)
                    XX[x] -= temp * parms.A[IDXA(x,i)];
#else
                    XX[x] -= temp * parms.A[IDXA(i,x)];
#endif
                }
            }
        }

        /* copy x from GRF back to GMEM */
        __syncthreads();
        addr = IDXX(tid);
        incr = XINC * parms.incx;
        for (i = tid; i < parms.n; i += XINC) {
            parms.x[addr] = XX[i];
            addr += incr;
        }
        __syncthreads();
    }
