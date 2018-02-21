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

    float cutLo = 4.441e-16f;
    float cutHi = 1.304e+19f;
    float sum, hiTest, t, ta, xmax, xmaxRecip;
    unsigned int i, state; 
    unsigned int n, tid, totalThreads, ctaStart;
    unsigned int ns;
    unsigned int totalIncx;
#if (USE_TEX==0)
    const float *sx;
#endif

    /* NOTE: wrapper must ensure that parms.n > 0 and parms.incx > 0 */
    tid = threadIdx.x;
    n = parms.n;
#if (USE_TEX==0)
    sx = parms.sx;
#endif
    totalThreads = gridDim.x * CUBLAS_SNRM2_THREAD_COUNT;
    ctaStart = CUBLAS_SNRM2_THREAD_COUNT * blockIdx.x;
    ns = n * parms.incx;
    totalIncx = totalThreads * parms.incx;

    hiTest = cutHi / (float)n;
    state = CUBLAS_SNRM2_STATE_ZERO;
    sum = 0.0f;
    xmax = 0.0f;
    t = 0.0f;
    ta = 0.0f;
    i = (ctaStart + tid) * parms.incx;
    while (state != CUBLAS_SNRM2_STATE_DONE) {
        /* we'd like a switch statement here */
        if (state == CUBLAS_SNRM2_STATE_ZERO) {
            while (i < ns) {
                if (!((t = fetchx(i)) == 0.0f)) {
                    break;
                }   
                i += totalIncx;
            }
            state = (i >= ns) ? CUBLAS_SNRM2_STATE_DONE : 
                                CUBLAS_SNRM2_STATE_TINY;
            continue;
        }
        if (state == CUBLAS_SNRM2_STATE_TINY) {
            xmax = fabsf(t);
            xmaxRecip = 1.0f / xmax;
            while (i < ns) {
                if (!((ta = fabsf(t = fetchx(i))) < cutLo)) {
                    break;
                }
                if (ta > xmax) {
                    /* Adjust scale factor */
                    t = xmax / t;
                    sum = 1.0f + sum * t * t;
                    xmax = ta;
                    xmaxRecip = 1.0f / xmax;
                } else {
                    t = t * xmaxRecip;
                    sum += t * t;
                }
                i += totalIncx;
            }
            if (i >= ns) {
                sum = xmax * sqrtf(sum);
                state = CUBLAS_SNRM2_STATE_DONE;
            } else {            
                state = CUBLAS_SNRM2_STATE_NORMAL;
            }
            continue;
        }
        if (state == CUBLAS_SNRM2_STATE_NORMAL) {
            sum = (sum * xmax) * xmax;
            while (i < ns) {
                if (!((ta = fabsf(t = fetchx(i))) < hiTest)){
                    break;
                }
                sum += t * t;
                i += totalIncx;
            }
            if (i >= ns) {
                sum = sqrtf(sum);
                state = CUBLAS_SNRM2_STATE_DONE;
            } else {
                state = CUBLAS_SNRM2_STATE_HUGE;
            }
            continue;
        }
        if (state == CUBLAS_SNRM2_STATE_HUGE) {
            xmax = ta;
            xmaxRecip = 1.0f / xmax;
            sum = (sum * xmaxRecip) * xmaxRecip;
            while (i < ns) {
                t = fetchx(i);
                ta = fabsf(t);
                if (ta > xmax) {
                    /* Adjust scale factor */
                    t = xmax / t;
                    sum = 1.0f + sum * t * t;
                    xmax = ta;
                    xmaxRecip = 1.0f / xmax;
                } else {
                    t = t * xmaxRecip;
                    sum += t * t;
                }
                i += totalIncx;
            }
            sum = xmax * sqrtf(sum);
            state = CUBLAS_SNRM2_STATE_DONE;
            continue;
        }
    }
    partialSum[tid] = sum;

    /*
     * FIXME: Because of the relatively complex state machine needed
     * to prevent overflow and underflow, right now we don't implement
     * a binary reduction tree but use a simple loop instead. Obviously
     * lower performance
     */

    __syncthreads();
    
    /* let thread 0 sum the partial dot products for this CTA */
    if (tid == 0) {
        int nbrSums = CUBLAS_SNRM2_THREAD_COUNT;
        i = 0;
        state = CUBLAS_SNRM2_STATE_ZERO;
        while (state != CUBLAS_SNRM2_STATE_DONE) {
            /* we'd like a switch statement here */
            if (state == CUBLAS_SNRM2_STATE_ZERO) {
                sum = 0.0f;
                while (i < nbrSums) {
                    if (!((t = partialSum[i]) == 0.0f)) {
                        break;
                    }
                    i++;
                }
                state = (i >= nbrSums) ? CUBLAS_SNRM2_STATE_DONE : 
                                         CUBLAS_SNRM2_STATE_TINY;
                continue;
            }
            if (state == CUBLAS_SNRM2_STATE_TINY) {
                xmax = fabsf(t);
                xmaxRecip = 1.0f / xmax;
                while (i < nbrSums) {
                    if (!((ta = fabsf(t = partialSum[i])) < cutLo)) {
                        break;
                    }
                    if (ta > xmax) {
                        /* Adjust scale factor */
                        t = xmax / t;
                        sum = 1.0f + sum * t * t;
                        xmax = ta;
                        xmaxRecip = 1.0f / xmax;
                    } else {
                        t = t * xmaxRecip;
                        sum += t * t;
                    }
                    i++; 
                }
                if (i >= nbrSums) {
                    sum = xmax * sqrtf(sum);
                    state = CUBLAS_SNRM2_STATE_DONE;
                } else {            
                    state = CUBLAS_SNRM2_STATE_NORMAL;
                }
                continue;
            }
            if (state == CUBLAS_SNRM2_STATE_NORMAL) {
                sum = (sum * xmax) * xmax;
                while (i < nbrSums) {
                    if (!((ta = fabsf(t = partialSum[i])) < hiTest)) {
                        break;
                    }
                    sum += t * t;
                    i++;  
                }
                if (i >= nbrSums) {
                    sum = sqrtf(sum);
                    state = CUBLAS_SNRM2_STATE_DONE;
                } else {
                    state = CUBLAS_SNRM2_STATE_HUGE;
                }
                continue;
            }
            if (state == CUBLAS_SNRM2_STATE_HUGE) {
                xmax = ta;
                xmaxRecip = 1.0f / xmax;
                sum = (sum * xmaxRecip) * xmaxRecip;
                while (i < nbrSums) {
                    t = partialSum[i];
                    ta = fabsf(t);
                    if (ta > xmax) {
                        /* Adjust scale factor */
                        t = xmax / t;
                        sum = 1.0f + sum * t * t;
                        xmax = ta;
                        xmaxRecip = 1.0f / xmax;
                    } else {
                        t = t * xmaxRecip;
                        sum += t * t;
                    }
                    i++; 
                }
                sum = xmax * sqrtf (sum);
                state = CUBLAS_SNRM2_STATE_DONE;
                continue;
            }
        }
        parms.result[blockIdx.x] = sum;
    }
