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

/* column-major ordering */
#if (FAST_IMUL==1)
#undef IMUL
#define IMUL(x,y)       __umul24(x,y)
#else
#undef IMUL
#define IMUL(x,y)       ((x)*(y))
#endif
#undef IDXA
#undef IDXX
#undef IDXY
#undef A_COL_OFS
#define IDXA(row,col)   (IMUL(parms.lda,(col))+(row))
#define IDXX(i)         (IMUL(parms.incx,(i))+startx)
#define IDXY(j)         (IMUL(parms.incy,(j))+starty)
#define A_COL_OFS       (parms.lda)

    unsigned int startx, starty, i, j, ii, jj, idx, tid, tidLo, tidHi;

    tid = threadIdx.x;
    tidLo = tid & (TILE_DIM - 1);
    tidHi = tid >> TILE_DIM_LOG;
    startx = (parms.incx >= 0) ? 0 : (IMUL((1 - parms.m),parms.incx));
    starty = (parms.incy >= 0) ? 0 : (IMUL((1 - parms.n),parms.incy));
#if (USE_MIXED_STEPPER==1)
    for (i = blockIdx.y * TILE_DIM; i < parms.m; i += SUP_TILE_DIM) {
        for (j = blockIdx.x * TILE_DIM; j < parms.n; j += SUP_TILE_DIM) {
            unsigned int ofs1, ofs2;
            /* syncthreads only needed if one CTA processes multiple tiles */
            __syncthreads();
#else
    {
        {
            unsigned int ofs1, ofs2;
            j = blockIdx.y * TILE_DIM;
            i = blockIdx.x * TILE_DIM;
#endif      
            /* read x[i] ... x[i+iinc-1] and y[j] ... y[j+jinc-1] */
            if (tid < TILE_DIM) {
                idx = i + tid;
                if (idx < parms.m) {
                    xi[tid] = parms.x[IDXX(idx)];
                }
                idx = j + tid;
                if (idx < parms.n) {
                    yj[tid] = parms.y[IDXY(idx)];
                }
            }
            __syncthreads();

            /* update elements in tile Ai,j */
            ofs1 = tidLo;
            ii = i + ofs1;
            ofs2 = tidHi;
            jj = j + ofs2;
            if ((ii < parms.m) && (jj < parms.n)) {
                idx = IDXA(ii,jj);
                parms.A[idx] += parms.alpha * xi[ofs1] * yj[ofs2];
#if (ELEMS_PER_THREAD >= 2)
                jj += A_NBR_COLS;
                if (jj < parms.n) {
                    idx += A_COL_OFS * A_NBR_COLS;
                    ofs2 += A_NBR_COLS;
                    parms.A[idx] += parms.alpha * xi[ofs1] * yj[ofs2];
#if (ELEMS_PER_THREAD >= 3)
                    jj += A_NBR_COLS;
                    if (jj < parms.n) {
                        idx += A_COL_OFS * A_NBR_COLS;
                        ofs2 += A_NBR_COLS;
                        parms.A[idx] += parms.alpha * xi[ofs1] * yj[ofs2];
#if (ELEMS_PER_THREAD >= 4)
                        jj += A_NBR_COLS;
                        if (jj < parms.n) {
                            idx += A_COL_OFS * A_NBR_COLS;
                            ofs2 += A_NBR_COLS;
                            parms.A[idx] += parms.alpha * xi[ofs1] * yj[ofs2];
#if (ELEMS_PER_THREAD >= 5)
#error ELEMS_PER_THREAD must be 1,2,3, or 4
#endif
                        }
#endif
                    }
#endif                            
                }                 
#endif
            }
        }
    }
