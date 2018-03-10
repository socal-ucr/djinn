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

#if (USE_TEX==1)
#undef fetchx
#define fetchx(i)  tex1Dfetch(texX,parms.texXOfs+(i))
#else
#undef fetchx
#define fetchx(i)  sx[i]
#endif /* USE_TEX */

    int i, n, tid, totalThreads, ctaStart;
    float *sx;

    /* NOTE: wrapper must ensure that parms.n > 0 and parms.incx > 0 */
    tid = threadIdx.x;
    n = parms.n;
    sx = parms.sx;
    totalThreads = gridDim.x * blockDim.x;
    ctaStart = blockDim.x * blockIdx.x;
    
    if (parms.incx == 1) {
        /* increment equal to 1 */
        for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
            sx[i] = fetchx(i) * parms.sa;
        }
    } else {
        /* increment not equal to 1 */
        int ix = ((parms.incx < 0) ? ((1 - n) * parms.incx) : 0);
        for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
            sx[ix+i*parms.incx] = fetchx(ix+i*parms.incx) * parms.sa;
        }
    }
