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

    unsigned int startx, i, j, ii, jj, x, tid;

    tid = threadIdx.x;
    startx = (parms.incx >= 0) ? 0 : ((1 - parms.n) * parms.incx);

#if (USE_MIXED_STEPPER == 1)
    for (i = blockIdx.y * TILE_DIM; i < parms.n; i += SUP_TILE_DIM) {
        for (j = blockIdx.x * TILE_DIM; j < parms.n; j += SUP_TILE_DIM) {
#if (LOWER==1)
            if (i < j) break;
#else
            if (j < i) continue;
#endif
#else /* USE_MIXED_STEPPER==1 */
    {
        {
            i = blockIdx.y * TILE_DIM;
            j = blockIdx.x * TILE_DIM;
#if (LOWER==1)
            if (i < j) return;
#else
            if (j < i) return;
#endif
            
#endif /* USE_MIXED_STEPPER==1 */

            /* read x[i] ... x[i+iinc-1] and x[j] ... x[j+jinc-1] */
            __syncthreads();
            if ((tid) < TILE_DIM) {
                int idx;
                idx = i + tid;
                if (idx < parms.n) {
                    xi[tid] = parms.x[IDXX(idx)];
                }
                idx = j + tid;
                if (idx < parms.n) {
                    xj[tid] = parms.x[IDXX(idx)];
                }
            }
            __syncthreads();

            /* update elements in tile Ai,j */
            ii = i + (tid & (TILE_DIM - 1));
            if (ii < parms.n) {
                jj = j + (tid >> TILE_DIM_LOG);
                for (x = 0; x < ELEMS_PER_THREAD; x++) {
#if (LOWER==1)
                    if ((jj < parms.n) && (ii >= jj)) {
#else 
                    if ((jj < parms.n) && (jj >= ii)) {
#endif
                        parms.A[IDXA(ii,jj)] += 
                            parms.alpha * xi[ii-i] * xj[jj-j];
                    }
                    jj += COL_INCR;
                }
            }
        }
    }

