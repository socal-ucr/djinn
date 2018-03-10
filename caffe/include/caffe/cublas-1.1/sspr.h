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

    unsigned int startx, ii, jj, iii, jjj, x, tid;

    tid = threadIdx.x;
    startx = (parms.incx >= 0) ? 0 : ((1 - parms.n) * parms.incx);

#if (LOWER==1)
    int twoN = parms.n << 1;
#endif
    for (ii = blockIdx.y * IIINC; ii < parms.n; ii += IINC) {
        for (jj = blockIdx.x * JJINC; jj < parms.n; jj += JINC) {
#if (LOWER==1)
            if (ii < jj) break;
#else
            if (jj < ii) continue;
#endif
            /* read x[ii] ... x[ii+iiinc-1] and x[jj] ... x[jj+jjinc-1] */
            __syncthreads();
            if ((tid) < BLK) {
                int idx;
                idx = ii + tid;
                if (idx < parms.n) {
                    xi[tid] = parms.x[IDXX(idx)];
                }
                idx = jj + tid;
                if (idx < parms.n) {
                    xj[tid] = parms.x[IDXX(idx)];
                }
            }
            __syncthreads();

            /* update elements in tile Aii,jj */
            iii = ii + (tid & (BLK - 1));
            if (iii < parms.n) {
                jjj = jj + (tid >> BLK_LOG);
                for (x = 0; x < ELEMS_PER_THREAD; x++) {
#if (LOWER==1)
                    if ((jjj < parms.n) && (iii >= jjj)) {
                        int idx = iii + (((twoN - jjj - 1) * jjj) >> 1);
#else
                    if ((jjj < parms.n) && (jjj >= iii)) {
                        int idx = iii + ((jjj * jjj + jjj) >> 1);
#endif
                        parms.AP[idx] += parms.alpha * xi[iii-ii] * xj[jjj-jj];
                    }
                    jjj += A_NBR_COLS;
                }
            }
        }
    }

