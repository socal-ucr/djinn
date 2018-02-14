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

#if (USE_TEX==1)
#undef fetchA
#undef fetchB
#define fetchA(i) tex1Dfetch(texA,(int)(parms.texAOfs+(i)))
#define fetchB(i) tex1Dfetch(texB,(int)(parms.texBOfs+(i)))
#else
#undef fetchA
#undef fetchB
#define fetchA(i) parms.A[i]
#define fetchB(i) parms.B[i]
#endif

#if (TILE_DIM==32)
#define ACCUMULATE_DOT_PRODUCT_TILE(num)    \
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
#define ACCUMULATE_2DOT_PRODUCTS_TILE(num1,num2,ljOfs)  \
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
#elif (TILE_DIM==16)
#define ACCUMULATE_DOT_PRODUCT_TILE(num)    \
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
} while (0)
#define ACCUMULATE_2DOT_PRODUCTS_TILE(num1,num2,ljOfs)  \
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
} while (0)
#else
#error TILE_DIM must be 16 or 32
#endif

#define ACCUMULATE_DOT_PRODUCT_N(num)       \
do {                                        \
    do {                                    \
        dp##num += (AA[li] * BB[lj]);       \
        li++;                               \
        lj++;                               \
        ll--;                               \
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
#if ((TRANSA==1)||(TRANSB==0))
    unsigned idxLoHi = IDXAA(tidLo,tidHi);
#endif
#if ((TRANSA==0)||(TRANSB==1))
    unsigned idxHiLo = IDXAA(tidHi,tidLo);
#endif


#if (USE_MIXED_STEPPER == 1)
    for (i = IMUL(blockIdx.y, TILE_DIM); i < parms.m; i += SUP_TILE_DIM) {
        unsigned ii_1 = i + tidLo;
#undef  ii_2
#define ii_2 (i + tidHi)  /* could be induction variable if enough registers */

        for (j = IMUL(blockIdx.x, TILE_DIM); j < parms.n; j += SUP_TILE_DIM) {
#undef  jj_2
#if ((TRANSB==0)&&(TRANSA==1))
            unsigned jj_2 = j + tidHi;
#else
#define jj_2 (j + tidHi)  /* could be induction variable if enough registers */
#endif
#else /* USE_MIXED_STEPPER==1 */
    {
        {
            i = IMUL(blockIdx.y, TILE_DIM);
            j = IMUL(blockIdx.x, TILE_DIM);
            unsigned ii_1 = i + tidLo;
#undef  ii_2
#if ((TRANSB==0)&&(TRANSA==1))
            unsigned ii_2 = i + tidHi;
#else
#define ii_2 (i + tidHi)  /* could be induction variable if enough registers */
#endif
#undef  jj_2
            unsigned jj_2 = j + tidHi;
#endif /* USE_MIXED_STEPPER==1 */


#undef  jj_1
#if ((TRANSB==1)&&(TRANSA==0))
            unsigned jj_1 = j + tidLo;
#else
#define jj_1 (j + tidLo)  /* could be induction variable if enough registers */
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
#if ((A_ELEMS_PER_THREAD >= 5)||(B_ELEMS_PER_THREAD >= 5))
                unsigned int offs2;
#endif

#undef ll_1
#undef ll_2
#if ((TRANSA==1)||(TRANSB==0))
#define ll_1 (l + tidLo) /* could be induction variable if enough registers */
#endif
#if ((TRANSA==0)||(TRANSB==1))
#define ll_2 (l + tidHi) /* could be induction variable if enough registers */
#endif
              
                /* Wait before clobbering old cache contents */
                __syncthreads ();
                
#if (TRANSA==0)
#if (A_ELEMS_PER_THREAD >= 5)
                ii = ii_1;
                IF (ii < parms.m) THEN
                    offs2 = tidHi;
                    for (ll = ll_2; ll < llLimit; ll += COL_INCR) {
                        AA[IDXAA(offs2,tidLo)] = fetchA(IDXA(ii,ll));
                        offs2 += COL_INCR;
                    }
                ENDIF
#else  /* A_ELEMS_PER_THREAD >= 5 */
#if (A_ELEMS_PER_THREAD >= 1)
                ll = ll_2;
                IF ((ii_1 < parms.m) && (ll < llLimit)) THEN
                    unsigned int idxAA;
                    unsigned int addrA;
                    idxAA = idxHiLo;
                    addrA = IDXA(ii_1,ll);
                    AA[idxAA] = fetchA(addrA);
#if (A_ELEMS_PER_THREAD >= 2)
                    ll += COL_INCR;
                    idxAA += COL_INCR;
                    addrA += COL_INCR * A_COL_OFS;
                    IF (ll < llLimit) THEN
                        AA[idxAA] = fetchA(addrA);
#if (A_ELEMS_PER_THREAD >= 3)
                        ll += COL_INCR;
                        idxAA += COL_INCR;
                        addrA += COL_INCR * A_COL_OFS;
                        IF (ll < llLimit) THEN
                            AA[idxAA] = fetchA(addrA);
#if (A_ELEMS_PER_THREAD >= 4)                               
                            ll += COL_INCR;
                            idxAA += COL_INCR;
                            addrA += COL_INCR * A_COL_OFS;
                            IF (ll < llLimit) THEN
                                AA[idxAA] = fetchA(addrA);
                            ENDIF
#endif  /* A_ELEMS_PER_THREAD >= 4 */
                        ENDIF
#endif  /* A_ELEMS_PER_THREAD >= 3 */
                    ENDIF
#endif  /* A_ELEMS_PER_THREAD >= 2 */
                ENDIF
#endif  /* A_ELEMS_PER_THREAD >= 1 */
#endif  /* A_ELEMS_PER_THREAD >= 5 */
#else   /* TRANSA = 0 */
#if (A_ELEMS_PER_THREAD >= 5)
                ll = ll_1;
                IF (ll < llLimit) THEN
                    unsigned int iiLimit = min (i + TILE_DIM, parms.m);
                    offs2 = tidHi;
                    for (ii = ii_2; ii < iiLimit; ii += COL_INCR) {
                        AA[IDXAA(tidLo,offs2)] = fetchA(IDXA(ll,ii));
                        offs2 += COL_INCR;
                    }
                ENDIF
#else  /* A_ELEMS_PER_THREAD >= 5 */
#if (A_ELEMS_PER_THREAD >= 1)
                ii = ii_2;
                IF ((ll_1 < llLimit) && (ii < parms.m)) THEN
                    unsigned int idxAA;
                    unsigned int addrA;
                    idxAA = idxLoHi;
                    addrA = IDXA(ll_1,ii);
                    AA[idxAA] = fetchA(addrA);
#if (A_ELEMS_PER_THREAD >= 2)
                    ii += COL_INCR;
                    idxAA += COL_INCR * AA_COL_OFS;
                    addrA += COL_INCR * A_COL_OFS;
                    IF (ii < parms.m) THEN
                        AA[idxAA] = fetchA(addrA);
#if (A_ELEMS_PER_THREAD >= 3)
                        ii += COL_INCR;
                        idxAA += COL_INCR * AA_COL_OFS;
                        addrA += COL_INCR * A_COL_OFS;                     
                        IF (ii < parms.m) THEN
                            AA[idxAA] = fetchA(addrA);
#if (A_ELEMS_PER_THREAD >= 4)                               
                            ii += COL_INCR;
                            idxAA += COL_INCR * AA_COL_OFS;
                            addrA += COL_INCR * A_COL_OFS;
                            IF (ii < parms.m) THEN
                                AA[idxAA] = fetchA(addrA);
                            ENDIF
#endif  /* A_ELEMS_PER_THREAD >= 4 */
                        ENDIF
#endif  /* A_ELEMS_PER_THREAD >= 3 */
                    ENDIF
#endif  /* A_ELEMS_PER_THREAD >= 2 */
                ENDIF
#endif  /* A_ELEMS_PER_THREAD >= 1 */
#endif  /* A_ELEMS_PER_THREAD >= 5 */
#endif  /* TRANSA = 0 */
                
#if (TRANSB==0)
#if (B_ELEMS_PER_THREAD >= 5)
                ll = ll_1;
                IF (ll < llLimit) THEN
                    unsigned int jjLimit = min (j + TILE_DIM, parms.n);
                    offs2 = tidHi;
                    for (jj = jj_2; jj < jjLimit; jj += COL_INCR) {
                        BB[IDXBB(tidLo,offs2)] = fetchB(IDXB(ll,jj));
                        offs2 += COL_INCR;
                    }
                ENDIF
#else /* B_ELEMS_PER_THREAD >= 5 */
#if (B_ELEMS_PER_THREAD >= 1)
                jj = jj_2;
                IF ((ll_1 < llLimit) && (jj < parms.n)) THEN
                    unsigned int idxBB;
                    unsigned int addrB;
                    idxBB = idxLoHi;
                    addrB = IDXB(ll_1,jj);
                    BB[idxBB] = fetchB(addrB);
#if (B_ELEMS_PER_THREAD >= 2)
                    jj += COL_INCR;
                    idxBB += COL_INCR * BB_COL_OFS;
                    addrB += COL_INCR * B_COL_OFS;
                    IF (jj < parms.n) THEN
                        BB[idxBB] = fetchB(addrB);
#if (B_ELEMS_PER_THREAD >= 3)
                        jj += COL_INCR;
                        idxBB += COL_INCR * BB_COL_OFS;
                        addrB += COL_INCR * B_COL_OFS; 
                        IF (jj < parms.n) THEN
                            BB[idxBB] = fetchB(addrB);
#if (B_ELEMS_PER_THREAD >= 4)
                            jj += COL_INCR;
                            idxBB += COL_INCR * BB_COL_OFS;
                            addrB += COL_INCR * B_COL_OFS; 
                            IF (jj < parms.n) THEN
                                BB[idxBB] = fetchB(addrB);
                            ENDIF
#endif  /* B_ELEMS_PER_THREAD >= 4 */
                        ENDIF                                    
#endif  /* B_ELEMS_PER_THREAD >= 3 */
                    ENDIF
#endif  /* B_ELEMS_PER_THREAD >= 2 */
                ENDIF
#endif  /* B_ELEMS_PER_THREAD >= 1 */
#endif  /* B_ELEMS_PER_THREAD >= 5 */
#else   /* TRANSB==0 */
#if (B_ELEMS_PER_THREAD >= 5)
                jj = jj_1;
                IF (jj < parms.n) THEN
                    offs2 = tidHi;
                    for (ll = ll_2; ll < llLimit; ll += COL_INCR) {
                        BB[IDXBB(offs2,tidLo)] = fetchB(IDXB(jj,ll));
                        offs2 += COL_INCR;
                    }
                ENDIF
#else  /* B_ELEMS_PER_THREAD >= 5 */
#if (B_ELEMS_PER_THREAD >= 1)
                ll = ll_2;
                IF ((jj_1 < parms.n) && (ll < llLimit)) THEN
                    unsigned int idxBB;
                    unsigned int addrB;
                    idxBB = idxHiLo;
                    addrB = IDXB(jj_1,ll);
                    BB[idxBB] = fetchB(addrB);
#if (B_ELEMS_PER_THREAD >= 2)
                    ll += COL_INCR;
                    idxBB += COL_INCR;
                    addrB += COL_INCR * B_COL_OFS;
                    IF (ll < llLimit) THEN
                        BB[idxBB] = fetchB(addrB);
#if (B_ELEMS_PER_THREAD >= 3)
                        ll += COL_INCR;
                        idxBB += COL_INCR;
                        addrB += COL_INCR * B_COL_OFS;
                        IF (ll < llLimit) THEN
                            BB[idxBB] = fetchB(addrB);
#if (B_ELEMS_PER_THREAD >= 4)
                            ll += COL_INCR;
                            idxBB += COL_INCR;
                            addrB += COL_INCR * B_COL_OFS;
                            IF (ll < llLimit) THEN
                                BB[idxBB] = fetchB(addrB);
                            ENDIF
#endif  /* B_ELEMS_PER_THREAD >= 4 */
                        ENDIF                                    
#endif  /* B_ELEMS_PER_THREAD >= 3 */
                    ENDIF
#endif  /* B_ELEMS_PER_THREAD >= 2 */
                ENDIF
#endif  /* B_ELEMS_PER_THREAD >= 1 */
#endif  /* B_ELEMS_PER_THREAD >= 5 */
#endif  /* TRANSB0==0 */
                
                /* Wait until new cache contents ready */
                __syncthreads ();
                
                /* We don't iterate over jj and ii since this is all done
                 * in parallel by the threads in each CTA.
                 */
                ii = tidLo;
                IF (ii < (parms.m - i)) THEN
                    unsigned int z = llLimit - l;
                    jj = tidHi;
                    IF (z == TILE_DIM) THEN
#if (C_ELEMS_PER_THREAD == 1)
                        IF (jj < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ACCUMULATE_DOT_PRODUCT_TILE(0);
                        ENDIF
#endif
#if (C_ELEMS_PER_THREAD == 2)
                        IF ((jj + COL_INCR) < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ACCUMULATE_2DOT_PRODUCTS_TILE(0,1,BB_COL_OFS*COL_INCR);
                        ELSEIF (jj < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ACCUMULATE_DOT_PRODUCT_TILE(0);
                        ENDIF
#endif
#if (C_ELEMS_PER_THREAD == 3)
                        IF ((jj + 2*COL_INCR) < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ACCUMULATE_2DOT_PRODUCTS_TILE(0,1,BB_COL_OFS*COL_INCR);
                            lj += 2 * BB_COL_OFS * COL_INCR;
                            ACCUMULATE_DOT_PRODUCT_TILE(2);
                        ELSEIF ((jj + COL_INCR) < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            CCUMULATE_2DOT_PRODUCTS_TILE(0,1,BB_COL_OFS*COL_INCR);
                        ELSEIF (jj < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ACCUMULATE_DOT_PRODUCT_TILE(0);
                        ENDIF
#endif
#if (C_ELEMS_PER_THREAD >= 4)
                        IF ((jj + 3*COL_INCR) < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ACCUMULATE_2DOT_PRODUCTS_TILE(0,1,BB_COL_OFS*COL_INCR);
                            lj += 2*BB_COL_OFS * COL_INCR;
                            ACCUMULATE_2DOT_PRODUCTS_TILE(2,3,BB_COL_OFS*COL_INCR);
                        ELSEIF ((jj + 2*COL_INCR) < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ACCUMULATE_2DOT_PRODUCTS_TILE(0,1,BB_COL_OFS*COL_INCR);
                            lj += 2*BB_COL_OFS * COL_INCR;
                            ACCUMULATE_DOT_PRODUCT_TILE(2);
                        ELSEIF ((jj + COL_INCR) < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ACCUMULATE_2DOT_PRODUCTS_TILE(0,1,BB_COL_OFS*COL_INCR);
                        ELSEIF (jj < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ACCUMULATE_DOT_PRODUCT_TILE(0);
                        ENDIF
#endif
#if (C_ELEMS_PER_THREAD >= 5)
#error C_ELEMS_PER_THREAD >= 5 no supported
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
#if (C_ELEMS_PER_THREAD == 3)
                        IF ((jj + 2*COL_INCR) < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(0);
                            jj += COL_INCR;
                            li = IDXAA(0,ii);
                            lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(1);
                            jj += COL_INCR;
                            li = IDXAA(0,ii);
                            lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(2);
                        ELSEIF ((jj + COL_INCR) < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(0);
                            li = IDXAA(0,ii);
                            jj += COL_INCR;
                            lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(1);
                        ELSEIF (jj < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(0);
                        ENDIF
#endif                        
#if (C_ELEMS_PER_THREAD == 4)
                        IF ((jj + 3*COL_INCR) < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(0);
                            jj += COL_INCR;
                            li = IDXAA(0,ii);
                            lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(1);
                            jj += COL_INCR;
                            li = IDXAA(0,ii);
                            lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(2);
                            jj += COL_INCR;
                            li = IDXAA(0,ii);
                            lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(3);
                        ELSEIF ((jj + 2*COL_INCR) < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(0);
                            jj += COL_INCR;
                            li = IDXAA(0,ii);
                            lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(1);
                            jj += COL_INCR;
                            li = IDXAA(0,ii);
                            lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(2);
                        ELSEIF ((jj + COL_INCR) < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(0);
                            li = IDXAA(0,ii);
                            jj += COL_INCR;
                            lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(1);
                        ELSEIF (jj < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N(0);
                        ENDIF
#endif
#if (C_ELEMS_PER_THREAD >= 5)
#error C_ELEMS_PER_THREAD >= 5 no supported
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
                ii = ii_1;
                jj = jj_2;
                IF ((ii < parms.m) && (jj < parms.n)) THEN
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
#endif  /* C_ELEMS_PER_THREAD >= 1 */
#endif
            } else {
#if (C_ELEMS_PER_THREAD >= 5)
                /* we would need an array dp[] instead of scalar dp0, .. */
#error C_ELEMS_PER_THREAD >= 5 no supported
#else
#if (C_ELEMS_PER_THREAD >= 1)
                ii = ii_1;
                jj = jj_2;
                IF ((ii < parms.m) && (jj < parms.n)) THEN
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
#endif  /* C_ELEMS_PER_THREAD >= 1 */
#endif
            }
        }
    }


