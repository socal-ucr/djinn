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
#endif

    int i, n, tid, totalThreads, ctaStart;
    float *sx;
    float *sy;

    /* NOTE: wrapper must ensure that parms.n > 0  */

    tid = threadIdx.x;
    n = parms.n;
    sx = parms.sx;
    sy = parms.sy;
    totalThreads = gridDim.x * blockDim.x;
    ctaStart = blockDim.x * blockIdx.x;

    if ((parms.incx == 0) || (parms.incy == 0)) {
        if ((blockIdx.x == 0) && (tid == 0)) {
            if ((parms.incx == 0) && (parms.incy == 0)) {
                if (parms.n & 1) {
                    float temp = sx[0];
                    sx[0] = sy[0];
                    sy[0] = temp;
                }
            } else if (parms.incx == 0) {
                int iy = (parms.incy < 0) ? ((1 - parms.n) * parms.incy) : 0;
                float oldTemp = sx[0];
                for (i = 0; i < parms.n; i++) {
                    float temp = sy[iy];
                    sy[iy] = oldTemp;
                    oldTemp = temp;
                    iy += parms.incy;
                }
                sx[0] = oldTemp;
            } else {
                int ix = (parms.incx < 0) ? ((1 - parms.n) * parms.incx) : 0;
                float oldTemp = parms.sy[0];
                for (i = 0; i < parms.n; i++) {
                    float temp = sx[ix];
                    sx[ix] = oldTemp;
                    oldTemp = temp;
                    ix += parms.incx;
                }
                sy[0] = oldTemp;
            }
        }
    } else if ((parms.incx == parms.incy) && (parms.incx > 0)) {
        /* equal, positive, increments */
        if (parms.incx == 1) {
            /* both increments equal to 1 */
            for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
#if (USE_TEX==0)
                float temp = sx[i];
                sx[i] = sy[i];
                sy[i] = temp;
#else
                /* ensure reads happen before writes */
                unsigned int t1 = __float_as_int(fetchx(i));
                unsigned int t2 = __float_as_int(fetchy(i));
                unsigned int t = t1 ^ t2;
                sx[i] = __int_as_float(t ^ t1);
                sy[i] = __int_as_float(t ^ t2);
#endif
            }
        } else {
            /* equal, positive, non-unit increments. */
            for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
#if (USE_TEX==0)
                float temp = sx[i*parms.incx];
                sx[i*parms.incx] = sy[i*parms.incx];
                sy[i*parms.incx] = temp;
#else
                /* ensure reads happen before writes */
                unsigned int t1 = __float_as_int(fetchx(i*parms.incx));
                unsigned int t2 = __float_as_int(fetchy(i*parms.incx));
                unsigned int t = t1 ^ t2;
                sx[i*parms.incx] = __int_as_float(t ^ t1);
                sy[i*parms.incx] = __int_as_float(t ^ t2);
#endif
            }
        }
    } else {
        /* unequal or nonpositive increments */
        int ix = ((parms.incx < 0) ? ((1 - n) * parms.incx) : 0);
        int iy = ((parms.incy < 0) ? ((1 - n) * parms.incy) : 0);
        for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
#if (USE_TEX==0)
            float temp = sx[ix+i*parms.incx];
            sx[ix+i*parms.incx] = sy[iy+i*parms.incy];
            sy[iy+i*parms.incy] = temp;
#else
            /* ensure reads happen before writes */
            unsigned int t1 = __float_as_int(fetchx(ix+i*parms.incx));
            unsigned int t2 = __float_as_int(fetchy(iy+i*parms.incy));
            unsigned int t = t1 ^ t2;
            sx[ix+i*parms.incx] = __int_as_float(t ^ t1);
            sy[iy+i*parms.incy] = __int_as_float(t ^ t2);
#endif
        }
    }
