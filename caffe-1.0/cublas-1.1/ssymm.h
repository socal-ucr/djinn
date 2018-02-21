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

/* Index functions for accessing surce/result matrices in GMEM, and cached
 * tiles in GRF. All matrices use column-major ordering. 
 */
#if (FAST_IMUL==1)
#undef IMUL
#define IMUL(x,y)       __umul24(x,y)
#else
#undef IMUL
#define IMUL(x,y)       ((x)*(y))
#endif

#define IDXA(row,col)   (IMUL(parms.lda,col)+(row)) /* index into matrix A */
#define IDXB(row,col)   (IMUL(parms.ldb,col)+(row)) /* index into matrix B */
#define IDXC(row,col)   (IMUL(parms.ldc,col)+(row)) /* index into matrix C */
#define IDXAA(row,col)  (__umul24(TILE_DIM+1,col)+(row)) /* index GRF A-tile */
#define IDXBB(row,col)  (__umul24(TILE_DIM+1,col)+(row)) /* index GRF B-tile */
#define AA_COL_OFS      (IDXAA(0,1)-IDXAA(0,0))
#define BB_COL_OFS      (IDXBB(0,1)-IDXBB(0,0))
#define A_COL_OFS       (IDXA(0,1)-IDXA(0,0))
#define B_COL_OFS       (IDXB(0,1)-IDXB(0,0))
#define C_COL_OFS       (IDXC(0,1)-IDXC(0,0))

#define ACCUMULATE_DOT_PRODUCT_32(num)      \
do {                                        \
    dp##num += (AA[li+ 0] * BB[lj+ 0]);     \
    dp##num += (AA[li+ 1] * BB[lj+ 1]);     \
    dp##num += (AA[li+ 2] * BB[lj+ 2]);     \
    dp##num += (AA[li+ 3] * BB[lj+ 3]);     \
    dp##num += (AA[li+ 4] * BB[lj+ 4]);     \
    dp##num += (AA[li+ 5] * BB[lj+ 5]);     \
    dp##num += (AA[li+ 6] * BB[lj+ 6]);     \
    dp##num += (AA[li+ 7] * BB[lj+ 7]);     \
    dp##num += (AA[li+ 8] * BB[lj+ 8]);     \
    dp##num += (AA[li+ 9] * BB[lj+ 9]);     \
    dp##num += (AA[li+10] * BB[lj+10]);     \
    dp##num += (AA[li+11] * BB[lj+11]);     \
    dp##num += (AA[li+12] * BB[lj+12]);     \
    dp##num += (AA[li+13] * BB[lj+13]);     \
    dp##num += (AA[li+14] * BB[lj+14]);     \
    dp##num += (AA[li+15] * BB[lj+15]);     \
    dp##num += (AA[li+16] * BB[lj+16]);     \
    dp##num += (AA[li+17] * BB[lj+17]);     \
    dp##num += (AA[li+18] * BB[lj+18]);     \
    dp##num += (AA[li+19] * BB[lj+19]);     \
    dp##num += (AA[li+20] * BB[lj+20]);     \
    dp##num += (AA[li+21] * BB[lj+21]);     \
    dp##num += (AA[li+22] * BB[lj+22]);     \
    dp##num += (AA[li+23] * BB[lj+23]);     \
    dp##num += (AA[li+24] * BB[lj+24]);     \
    dp##num += (AA[li+25] * BB[lj+25]);     \
    dp##num += (AA[li+26] * BB[lj+26]);     \
    dp##num += (AA[li+27] * BB[lj+27]);     \
    dp##num += (AA[li+28] * BB[lj+28]);     \
    dp##num += (AA[li+29] * BB[lj+29]);     \
    dp##num += (AA[li+30] * BB[lj+30]);     \
    dp##num += (AA[li+31] * BB[lj+31]);     \
} while (0)

#define ACCUMULATE_2DOT_PRODUCTS_32(num1,num2,ljOfs)  \
do {                                                  \
    dp##num1 += (AA[li+ 0] * BB[lj+ 0]);              \
    dp##num2 += (AA[li+ 0] * BB[lj+(ljOfs)+ 0]);      \
    dp##num1 += (AA[li+ 1] * BB[lj+ 1]);              \
    dp##num2 += (AA[li+ 1] * BB[lj+(ljOfs)+ 1]);      \
    dp##num1 += (AA[li+ 2] * BB[lj+ 2]);              \
    dp##num2 += (AA[li+ 2] * BB[lj+(ljOfs)+ 2]);      \
    dp##num1 += (AA[li+ 3] * BB[lj+ 3]);              \
    dp##num2 += (AA[li+ 3] * BB[lj+(ljOfs)+ 3]);      \
    dp##num1 += (AA[li+ 4] * BB[lj+ 4]);              \
    dp##num2 += (AA[li+ 4] * BB[lj+(ljOfs)+ 4]);      \
    dp##num1 += (AA[li+ 5] * BB[lj+ 5]);              \
    dp##num2 += (AA[li+ 5] * BB[lj+(ljOfs)+ 5]);      \
    dp##num1 += (AA[li+ 6] * BB[lj+ 6]);              \
    dp##num2 += (AA[li+ 6] * BB[lj+(ljOfs)+ 6]);      \
    dp##num1 += (AA[li+ 7] * BB[lj+ 7]);              \
    dp##num2 += (AA[li+ 7] * BB[lj+(ljOfs)+ 7]);      \
    dp##num1 += (AA[li+ 8] * BB[lj+ 8]);              \
    dp##num2 += (AA[li+ 8] * BB[lj+(ljOfs)+ 8]);      \
    dp##num1 += (AA[li+ 9] * BB[lj+ 9]);              \
    dp##num2 += (AA[li+ 9] * BB[lj+(ljOfs)+ 9]);      \
    dp##num1 += (AA[li+10] * BB[lj+10]);              \
    dp##num2 += (AA[li+10] * BB[lj+(ljOfs)+10]);      \
    dp##num1 += (AA[li+11] * BB[lj+11]);              \
    dp##num2 += (AA[li+11] * BB[lj+(ljOfs)+11]);      \
    dp##num1 += (AA[li+12] * BB[lj+12]);              \
    dp##num2 += (AA[li+12] * BB[lj+(ljOfs)+12]);      \
    dp##num1 += (AA[li+13] * BB[lj+13]);              \
    dp##num2 += (AA[li+13] * BB[lj+(ljOfs)+13]);      \
    dp##num1 += (AA[li+14] * BB[lj+14]);              \
    dp##num2 += (AA[li+14] * BB[lj+(ljOfs)+14]);      \
    dp##num1 += (AA[li+15] * BB[lj+15]);              \
    dp##num2 += (AA[li+15] * BB[lj+(ljOfs)+15]);      \
    dp##num1 += (AA[li+16] * BB[lj+16]);              \
    dp##num2 += (AA[li+16] * BB[lj+(ljOfs)+16]);      \
    dp##num1 += (AA[li+17] * BB[lj+17]);              \
    dp##num2 += (AA[li+17] * BB[lj+(ljOfs)+17]);      \
    dp##num1 += (AA[li+18] * BB[lj+18]);              \
    dp##num2 += (AA[li+18] * BB[lj+(ljOfs)+18]);      \
    dp##num1 += (AA[li+19] * BB[lj+19]);              \
    dp##num2 += (AA[li+19] * BB[lj+(ljOfs)+19]);      \
    dp##num1 += (AA[li+20] * BB[lj+20]);              \
    dp##num2 += (AA[li+20] * BB[lj+(ljOfs)+20]);      \
    dp##num1 += (AA[li+21] * BB[lj+21]);              \
    dp##num2 += (AA[li+21] * BB[lj+(ljOfs)+21]);      \
    dp##num1 += (AA[li+22] * BB[lj+22]);              \
    dp##num2 += (AA[li+22] * BB[lj+(ljOfs)+22]);      \
    dp##num1 += (AA[li+23] * BB[lj+23]);              \
    dp##num2 += (AA[li+23] * BB[lj+(ljOfs)+23]);      \
    dp##num1 += (AA[li+24] * BB[lj+24]);              \
    dp##num2 += (AA[li+24] * BB[lj+(ljOfs)+24]);      \
    dp##num1 += (AA[li+25] * BB[lj+25]);              \
    dp##num2 += (AA[li+25] * BB[lj+(ljOfs)+25]);      \
    dp##num1 += (AA[li+26] * BB[lj+26]);              \
    dp##num2 += (AA[li+26] * BB[lj+(ljOfs)+26]);      \
    dp##num1 += (AA[li+27] * BB[lj+27]);              \
    dp##num2 += (AA[li+27] * BB[lj+(ljOfs)+27]);      \
    dp##num1 += (AA[li+28] * BB[lj+28]);              \
    dp##num2 += (AA[li+28] * BB[lj+(ljOfs)+28]);      \
    dp##num1 += (AA[li+29] * BB[lj+29]);              \
    dp##num2 += (AA[li+29] * BB[lj+(ljOfs)+29]);      \
    dp##num1 += (AA[li+30] * BB[lj+30]);              \
    dp##num2 += (AA[li+30] * BB[lj+(ljOfs)+30]);      \
    dp##num1 += (AA[li+31] * BB[lj+31]);              \
    dp##num2 += (AA[li+31] * BB[lj+(ljOfs)+31]);      \
} while (0)

#define ACCUMULATE_DOT_PRODUCT_N(num)       \
do {                                        \
    do {                                    \
        dp##num += (AA[li++] * BB[lj++]);   \
        ll-= 1;                             \
    } while (ll);                           \
} while (0)

#define ACCUMULATE_2DOT_PRODUCTS_N(num1,num2,ljOfs)  \
do {                                                 \
    do {                                             \
        dp##num1 += (AA[li+ 0] * BB[lj+ 0]);         \
        dp##num2 += (AA[li+ 0] * BB[lj+(ljOfs)+ 0]); \
        li++;                                        \
        lj++;                                        \
        ll--;                                        \
    } while(ll);                                     \
} while (0)

#undef IF
#undef THEN
#undef ENDIF
#undef ELSE
#undef ELSEIF
#if FULL_TILES_ONLY==1
#define IF(x)
#define THEN       {
#define ENDIF      }
#define ELSE       } if (0) {
#define ELSEIF(x)  } if (0)
#else
#define IF(x)      if (x)
#define THEN       {
#define ENDIF      }
#define ELSE       } else {
#define ELSEIF(x)  } else if (x)
#endif

    unsigned int i, j, l, ii, jj, ll, tid = threadIdx.x;
#if (C_ELEMS_PER_THREAD >= 1)
    float dp0;
#if (C_ELEMS_PER_THREAD >= 2)
    float dp1;
#if (C_ELEMS_PER_THREAD >= 3)
    float dp2;
#if (C_ELEMS_PER_THREAD >= 4)
    float dp3;
#if (C_ELEMS_PER_THREAD >= 5)
#error C_ELEMS_PER_THREAD >= 5 not supported
#endif /* (C_ELEMS_PER_THREAD >= 5) */
#endif /* (C_ELEMS_PER_THREAD >= 4) */
#endif /* (C_ELEMS_PER_THREAD >= 3) */
#endif /* (C_ELEMS_PER_THREAD >= 2) */
#endif /* (C_ELEMS_PER_THREAD >= 1) */
    unsigned tidLo = (tid & (TILE_DIM - 1));
    unsigned tidHi = (tid >> TILE_DIM_LOG);
#if (USE_MIXED_STEPPER == 1)
    for (i = IMUL(blockIdx.y,TILE_DIM); i < parms.m; i += SUP_TILE_DIM) {
       for (j = IMUL(blockIdx.x,TILE_DIM); j < parms.n; j += SUP_TILE_DIM) {
#else
    {
        {
            i = IMUL(blockIdx.y,TILE_DIM);
            j = IMUL(blockIdx.x,TILE_DIM);
#endif
            /* set accumulation to 0*/
#if (C_ELEMS_PER_THREAD >= 5)
#error C_ELEMS_PER_THREAD >= 5 not supported
#else
#if (C_ELEMS_PER_THREAD >= 1)
            dp0 = 0.0f;
#if (C_ELEMS_PER_THREAD >= 2)
            dp1 = 0.0f;
#if (C_ELEMS_PER_THREAD >= 3)
            dp2 = 0.0f;
#if (C_ELEMS_PER_THREAD >= 4)
            dp3 = 0.0f;
#endif  /* C_ELEMS_PER_THREAD >= 4 */
#endif  /* C_ELEMS_PER_THREAD >= 3 */
#endif  /* C_ELEMS_PER_THREAD >= 2 */
#endif  /* C_ELEMS_PER_THREAD >= 1 */
#endif  /* C_ELEMS_PER_THREAD >= 5 */

            for (l = 0; l < parms.k; l += TILE_DIM) {
                unsigned int llLimit = min ((l + TILE_DIM), parms.k);
                unsigned int z = llLimit - l;
                unsigned int offs2;
                
                /* Wait before clobbering old cache contents */
                __syncthreads ();
                
                ii = i + tidLo;
                IF (ii < parms.m) THEN
                    offs2 = tidHi;
#if (LSIDE==1)
#if (UPPER==1)
                    for (ll = l+offs2; ll<llLimit; ll += COL_INCR) {
                        unsigned int addr = (ii<=ll)?IDXA(ii,ll):IDXA(ll,ii);
                        AA[IDXAA(offs2,tidLo)] = parms.A[addr];
                        offs2 += COL_INCR;
                    }
#else
                    for (ll = l+offs2; ll<llLimit; ll += COL_INCR) {
                        unsigned int addr = (ii>=ll)?IDXA(ii,ll):IDXA(ll,ii);
                        AA[IDXAA(offs2,tidLo)] = parms.A[addr];
                        offs2 += COL_INCR;
                    }
#endif
#else /* LSIDE==1 */
                    for (ll = l + offs2; ll < llLimit; ll += COL_INCR) {
                        AA[IDXAA(offs2,tidLo)] = parms.A[IDXA(ii,ll)];
                        offs2 += COL_INCR;
                    }
#endif /* LSIDE==1 */
                ENDIF

                ll = l + tidLo;
                IF (ll < llLimit) THEN
                    unsigned int jjLimit = min (j + TILE_DIM, parms.n);
                    offs2 = tidHi;
#if (LSIDE==1)
                    for (jj = j + offs2; jj < jjLimit; jj += COL_INCR) {
                        BB[IDXBB(tidLo, offs2)] = parms.B[IDXB(ll,jj)];
                        offs2 += COL_INCR;
                    }
#else
#if (UPPER==1)
                    for (jj = j + offs2; jj < jjLimit; jj += COL_INCR) {
                        unsigned int addr = (ll<=jj)?IDXB(ll,jj):IDXB(jj,ll);
                        BB[IDXBB(tidLo, offs2)] = parms.B[addr];
                        offs2 += COL_INCR;
                    }
#else
                    for (jj = j + offs2; jj < jjLimit; jj += COL_INCR) {
                        unsigned int addr = (jj<=ll)?IDXB(ll,jj):IDXB(jj,ll);
                        BB[IDXBB(tidLo, offs2)] = parms.B[addr];
                        offs2 += COL_INCR;
                    }
#endif
#endif /* LSIDE==1 */
                ENDIF

                /* Wait until new cache contents ready */
                __syncthreads ();
                
                /* We don't iterate over jj and ii since this is al done
                 * in paralel by the threads in each CTA.
                 */
                ii = tidLo;
                IF (ii < (parms.m - i)) THEN
                    jj = tidHi;
                    IF (z == 32) THEN
#if (C_ELEMS_PER_THREAD == 1)
                        IF (jj < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ACCUMULATE_DOT_PRODUCT_32(0);
                        ENDIF
#endif
#if (C_ELEMS_PER_THREAD == 2)
                        IF ((jj + COL_INCR) < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ACCUMULATE_2DOT_PRODUCTS_32(0,1,BB_COL_OFS*COL_INCR);
                        ELSEIF (jj < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ACCUMULATE_DOT_PRODUCT_32(0);
                        ENDIF
#endif
#if (C_ELEMS_PER_THREAD >= 3)
#error C_ELEMS_PER_THREAD >= 3 no supported
#endif
                    ELSE                        
#if (C_ELEMS_PER_THREAD == 1)
                        IF (jj < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(0);
                        ENDIF
#endif
#if (C_ELEMS_PER_THREAD == 2)
                        IF ((jj + COL_INCR) < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_2DOT_PRODUCTS_N(0,1,BB_COL_OFS*COL_INCR);
                        ELSEIF (jj < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(0);
                        ENDIF
#endif
#if (C_ELEMS_PER_THREAD >= 3)
#error C_ELEMS_PER_THREAD >= 3 no supported
#endif
                    ENDIF
                ENDIF
            }
            /* write out completed tile of matrix C */
            if (parms.beta == 0.0f) {
#if (C_ELEMS_PER_THREAD >= 5)
                /* we would need an array dp[] instead of scalar dp0, .. */
#error C_ELEMS_PER_THREAD >= 5 no supported
#else
#if (C_ELEMS_PER_THREAD >= 1)
                ii = i + tidLo;
                IF (ii < parms.m) THEN
                    jj = j + tidHi;
                    IF (jj < parms.n) THEN
                        unsigned int addrC = IDXC(ii,jj);
                        parms.C[addrC] = parms.alpha * dp0;
#if (C_ELEMS_PER_THREAD >= 2)
                        jj += COL_INCR;
                        IF (jj < parms.n) THEN
                            addrC += COL_INCR * C_COL_OFS;
                            parms.C[addrC] = parms.alpha * dp1;
#if (C_ELEMS_PER_THREAD >= 3)
                            jj += COL_INCR;
                            IF (jj < parms.n) THEN
                                addrC += COL_INCR * C_COL_OFS;
                                parms.C[addrC] = parms.alpha * dp2;
#if (C_ELEMS_PER_THREAD >= 4)
                                jj += COL_INCR;
                                IF (jj < parms.n) THEN
                                    addrC += COL_INCR * C_COL_OFS;
                                    parms.C[addrC] = parms.alpha * dp3;
                                ENDIF
#endif  /* C_ELEMS_PER_THREAD >= 4 */
                            ENDIF
#endif  /* C_ELEMS_PER_THREAD >= 3 */
                        ENDIF
#endif  /* C_ELEMS_PER_THREAD >= 2 */
                    ENDIF
                ENDIF
#endif  /* C_ELEMS_PER_THREAD >= 1 */
#endif
            } else {
#if (C_ELEMS_PER_THREAD >= 5)
                /* we would need an array dp[] instead of scalar dp0, .. */
#error C_ELEMS_PER_THREAD >= 5 no supported
#else
#if (C_ELEMS_PER_THREAD >= 1)
                ii = i + tidLo;
                IF (ii < parms.m) THEN
                    jj = j + tidHi;
                    IF (jj < parms.n) THEN
                        unsigned int addrC = IDXC(ii,jj);
                        parms.C[addrC] = parms.beta * parms.C[addrC] + parms.alpha * dp0;
#if (C_ELEMS_PER_THREAD >= 2)
                        jj += COL_INCR;
                        IF (jj < parms.n) THEN
                            addrC += COL_INCR * C_COL_OFS;
                            parms.C[addrC] = parms.beta * parms.C[addrC] + parms.alpha * dp1;
#if (C_ELEMS_PER_THREAD >= 3)
                            jj += COL_INCR;
                            IF (jj < parms.n) THEN
                                addrC += COL_INCR * C_COL_OFS;
                                parms.C[addrC] = parms.beta * parms.C[addrC] + parms.alpha * dp2;
#if (C_ELEMS_PER_THREAD >= 4)
                                jj += COL_INCR;
                                IF (jj < parms.n) THEN
                                    addrC += COL_INCR * C_COL_OFS;
                                    parms.C[addrC] = parms.beta * parms.C[addrC] + parms.alpha * dp3;
                                ENDIF
#endif  /* C_ELEMS_PER_THREAD >= 4 */
                            ENDIF
#endif  /* C_ELEMS_PER_THREAD >= 3 */
                        ENDIF
#endif  /* C_ELEMS_PER_THREAD >= 2 */
                    ENDIF
                ENDIF
#endif  /* C_ELEMS_PER_THREAD >= 1 */
#endif
            }
        }
    }


