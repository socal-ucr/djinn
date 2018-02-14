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
#undef fetchy
#define fetchx(i)  tex1Dfetch(texX,parms.texXOfs+(i))
#define fetchy(i)  tex1Dfetch(texY,parms.texYOfs+(i))
#else
#undef fetchx
#undef fetchy
#define fetchx(i)  sx[i]
#define fetchy(i)  sy[i]
#endif /* USE_TEX */

    int i, n, tid, totalThreads, ctaStart;
#if (USE_TEX==0)
    const float *sx;
#endif
    float *sy;
    
    /* NOTE: wrapper must ensure that parms.n > 0  */
    tid = threadIdx.x;
    n = parms.n;
#if (USE_TEX==0)
    sx = parms.sx;
#endif
    sy = parms.sy;

    totalThreads = gridDim.x*blockDim.x;
    ctaStart = blockDim.x*blockIdx.x;
   
    if (parms.incy == 0) {
        if ((blockIdx.x == 0) && (tid == 0)) {
            /* FIXME: This code is functionally correct, but inefficient */
            int ix = (parms.incx < 0) ? ((1 - parms.n) * parms.incx) : 0;
            float sum = 0.0f;
            for (i = 0; i < parms.n; i++) {
                sum += parms.sa * fetchx(ix);
                ix += parms.incx;
            }
            sy[0] += sum;
        }
    } else if ((parms.incx == parms.incy) && (parms.incx > 0)) {
        /* equal, positive, increments */
        if (parms.incx == 1) {
            /* both increments equal to 1 */
            for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
                sy[i] = fetchy(i) + parms.sa * fetchx(i);
            }
        } else {
            /* equal, positive, non-unit increments. */
            for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
                sy[i*parms.incx] = (fetchy(i*parms.incx) + 
                                    parms.sa * fetchx(i*parms.incx));
            }
        }
    } else {
        /* unequal or nonpositive increments */
        int ix = ((parms.incx < 0) ? ((1 - n) * parms.incx) : 0);
        int iy = ((parms.incy < 0) ? ((1 - n) * parms.incy) : 0);
        for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
            sy[iy+i*parms.incy] = (fetchy(iy+i*parms.incy) + 
                                   parms.sa * fetchx(ix+i*parms.incx));
        }
    }

