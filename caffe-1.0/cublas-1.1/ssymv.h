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

    int i, ii, j, jj, idx, incr, tid;
    float sdot;
    int startx;
    int starty;

    /*
     * NOTE: wrapper must ensure that parms.n >= 0, and that parms.incx and 
     *       parms.incy are != 0 
     */

    tid = threadIdx.x;
    startx = (parms.incx >= 0) ? 0 : ((1 - parms.n) * parms.incx);
    starty = (parms.incy >= 0) ? 0 : ((1 - parms.n) * parms.incy);

    /* step CTA array over the rows */
    for (i = 0; i < parms.n; i += IINC) {
        /* first row being processed by this CTA */
        ii = i + blockIdx.x * CUBLAS_SSYMV_THREAD_COUNT;            
        if (ii >= parms.n) break; /* nothing to do for this CTA */
        ii += tid; /* row being processed by this thread */
        sdot = 0.0f; /* initialize dot product handled by this thread */
        /* iterate over chunks of rows. These chunks are very large, so
         * in many case we'll only executed the loop body once, i.e. we'll
         * process the whole row in one fell swoop.
         */
        for (j = 0; j < parms.n; j += JINC) {
            int jjLimit = min (j + JINC, parms.n);
            incr = XINC * parms.incx;
            jj = j + tid;
            __syncthreads ();
            idx = IDXX(jj);
#if (X_ELEMS_PER_THREAD == 4)
            if (jj < (jjLimit - 3 * XINC)) {
                XX[tid+0*XINC] = parms.alpha * parms.x[idx + 0 * incr];
                XX[tid+1*XINC] = parms.alpha * parms.x[idx + 1 * incr];
                XX[tid+2*XINC] = parms.alpha * parms.x[idx + 2 * incr];
                XX[tid+3*XINC] = parms.alpha * parms.x[idx + 3 * incr];
            }
            else if (jj < (jjLimit - 2 * XINC)) {
                XX[tid+0*XINC] = parms.alpha * parms.x[idx + 0 * incr];
                XX[tid+1*XINC] = parms.alpha * parms.x[idx + 1 * incr];
                XX[tid+2*XINC] = parms.alpha * parms.x[idx + 2 * incr];
            }
            else if (jj < (jjLimit - 1 * XINC)) {
                XX[tid+0*XINC] = parms.alpha * parms.x[idx + 0 * incr];
                XX[tid+1*XINC] = parms.alpha * parms.x[idx + 1 * incr];
            }
            else if (jj < (jjLimit - 0 * XINC)) {
                XX[tid+0*XINC] = parms.alpha * parms.x[idx + 0 * incr];
            }
#else
#error current code cannot handle X_ELEMS_PER_THREAD != 4
#endif
            __syncthreads ();
            
            if (ii < parms.n) { /* if this row is active, accumulate dp */
                jj = j;
                while (jj < jjLimit) {
                    float temp;
#if (UPPER==1)
                    temp = (ii <= jj) ? 
                        parms.A[IDXA(ii,jj)] : parms.A[IDXA(jj,ii)];
#else
                    temp = (ii >= jj) ? 
                        parms.A[IDXA(ii,jj)] : parms.A[IDXA(jj,ii)];
#endif
                    sdot += temp * XX[jj-j];
                    jj++;
                }
            }
        }
        if (ii < parms.n) { /* if this row is active, write out dp */
            idx = IDXY(ii);
            if (parms.beta != 0.0f) {
                sdot += parms.beta * parms.y[idx];
            }
            parms.y[idx] = sdot;
        }
    }
