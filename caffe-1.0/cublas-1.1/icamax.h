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
#define fetchx(i)  cx[i]
#endif /* USE_TEX */

    int i, jmax, tid, totalThreads, ctaStart;
    int imax = 0;
    float cmax = 0.0f;
    float xabs;
#if (USE_TEX==0)
    const cuComplex *cx;
#endif

    /* NOTE: wrapper must ensure that parms.n > 0 and parms.incx > 0 */

    tid = threadIdx.x;
#if (USE_TEX==0)
    cx = parms.cx;
#endif
    totalThreads = gridDim.x * CUBLAS_ICAMAX_THREAD_COUNT;
    ctaStart = CUBLAS_ICAMAX_THREAD_COUNT * blockIdx.x;    

    if (parms.incx == 1) {
         /* increment equal to 1 */
        for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
            cuComplex tx = fetchx(i);
            xabs = fabsf (cuCrealf (tx)) + fabsf (cuCimagf (tx));
            if (xabs > cmax) {
                imax = i;
                cmax = xabs;
            }
        }
    } else {
        /* increment not equal to 1 */
        for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
            cuComplex tx = fetchx(i*parms.incx);
            xabs = fabsf (cuCrealf (tx)) + fabsf (cuCimagf (tx));
            if (xabs > cmax) {
                imax = i;
                cmax = xabs;
            }
        }
    }
    partialIMax[tid] = imax;
    partialCMax[tid] = cmax;

#if (CUBLAS_ICAMAX_THREAD_COUNT & (CUBLAS_ICAMAX_THREAD_COUNT - 1))
#error code requires CUBLAS_ICAMAX_THREAD_COUNT to be a power of 2
#endif

    for (i = CUBLAS_ICAMAX_THREAD_COUNT >> 1; i > 0; i >>= 1) {
        __syncthreads(); 
        if (tid < i) {
            cmax = partialCMax[tid];
            imax = partialIMax[tid];
            xabs = partialCMax[tid + i];
            jmax = partialIMax[tid + i];
            if ((xabs > cmax) || ((xabs == cmax) && (jmax < imax))) {
                imax = jmax;
                cmax = xabs;
            }
            partialCMax[tid] = cmax;
            partialIMax[tid] = imax;
        }
    }
    if (tid == 0) {
        parms.resMax[blockIdx.x] = partialCMax[tid];
        parms.resPos[blockIdx.x] = partialIMax[tid];
    }
