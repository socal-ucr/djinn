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

#if (FAST_IMUL==1)
#undef IMUL
#define IMUL(x,y)       __umul24(x,y)
#else
#undef IMUL
#define IMUL(x,y)       ((x)*(y))
#endif

/* Index functions for accessing surce/result matrices in GMEM, and cached
 * tiles in GRF. All matrices use column-major ordering. 
 */
#define IDXA(row,col)  (IMUL(parms.lda,col)+(row)) /* index into matrix A */
#define IDXB(row,col)  (IMUL(parms.ldb,col)+(row)) /* index into matrix B */
#define IDXC(row,col)  (IMUL(parms.ldc,col)+(row)) /* index into matrix C */
#define IDXAA(row,col) (__umul24(TILE_DIM+1,col)+(row)) /* index GRF A-tile */
#define IDXBB(row,col) (__umul24(TILE_DIM+1,col)+(row)) /* index GRF B-tile */
#define AA_COL_OFS     (IDXAA(0,1)-IDXAA(0,0))
#define BB_COL_OFS     (IDXBB(0,1)-IDXBB(0,0))
#define A_COL_OFS      (IDXA(0,1)-IDXA(0,0))
#define B_COL_OFS      (IDXB(0,1)-IDXB(0,0))
#define C_COL_OFS      (IDXC(0,1)-IDXC(0,0))


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
    while (ll) {                            \
        dp##num += (AA[li++] * BB[lj++]);   \
        ll -= 1;                            \
    }                                       \
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
    
#if ((C_ELEMS_PER_THREAD >= 5) || (A_ELEMS_PER_THREAD >= 5))
    unsigned int x;
#endif

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
    unsigned int tidLo = (tid & (TILE_DIM - 1));
    unsigned int tidHi = (tid >> TILE_DIM_LOG);  

#if (USE_MIXED_STEPPER == 1)
    for (i = IMUL(blockIdx.y, TILE_DIM); i < parms.n; i += SUP_TILE_DIM) {
       for (j = IMUL(blockIdx.x, TILE_DIM); j < parms.n; j += SUP_TILE_DIM) {
#if ((A_ELEMS_PER_THREAD >= 3)||(B_ELEMS_PER_THREAD >= 3))
           unsigned int offs1, offs2;
#endif
#if (UPPER==1)
           if (i > j) continue;
#else
           if (j > i) break;
#endif
#else /* USE_MIXED_STEPPER */
    {
        {
#if ((A_ELEMS_PER_THREAD >= 3)||(B_ELEMS_PER_THREAD >= 3))            
            unsigned int offs1, offs2;
#endif
            i = IMUL(blockIdx.y, TILE_DIM);
            j = IMUL(blockIdx.x, TILE_DIM);
#if (UPPER==1)
            if (i > j) return;
#else
            if (j > i) return;
#endif
            
#endif /* USE_MIXED_STEPPER */ 
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
                /* Wait until all threads are done using the previously cached
                 * pair of tiles of source matrices A and B
                 */
                __syncthreads ();

                /* Copy a tile from source matrix A. Only tile elements that 
                 * represent actual elements of matrix A are copied. Elements
                 * that lie outside the source matrix (as determined by check
                 * against iiiLimit and lllLimit) are not touched. Since reads
                 * from GMEM are the 'expensive' part of this cpde, throw in 
                 * the required multiply by alpha, since it's essentially free.
                 * During subsequent dot product computations, we conceptually
                 * multiply rows of the A-tile (called AA) by columns of the
                 * B-tile (called BB). Since the preferred access pattern when
                 * using column-major storage is access a 2D-matrix in columns
                 * we transpose A-tile by default, so dot product computation
                 * walks the columns of both A and B tiles. Any transposition
                 * requested by caller is superimposed on this transposition.
                 */               
#if (TRANSA==0)
#if (A_ELEMS_PER_THREAD >= 3)
                ii = i + tidLo;
                IF (ii < parms.n) THEN
                    offs2 = tidHi;
                    for (ll = l + offs2; ll < llLimit; ll += COL_INCR) {
                        AA[IDXAA(offs2,tidLo)] = parms.A[IDXA(ii,ll)]);
                        offs2 += COL_INCR;
                    }
                ENDIF
#else  /* A_ELEMS_PER_THREAD >= 3 */
#if (A_ELEMS_PER_THREAD >= 1)
                ii = i + tidLo;
                IF (ii < parms.n) THEN
                    ll = l + tidHi;
                    IF (ll < llLimit) THEN
                        unsigned int idxAA;
                        unsigned int addrA;
                        idxAA = IDXAA(tidHi,tidLo);
                        addrA = IDXA(ii,ll);
                        AA[idxAA] = parms.A[addrA];
#if (A_ELEMS_PER_THREAD >= 2)
                        ll += COL_INCR;
                        IF (ll < llLimit) THEN
                            idxAA += COL_INCR;
                            addrA += COL_INCR * A_COL_OFS;
                            AA[idxAA] = parms.A[addrA];
                        ENDIF
#endif  /* A_ELEMS_PER_THREAD >= 2 */
                    ENDIF
                ENDIF
#endif  /* A_ELEMS_PER_THREAD >= 1 */
#endif  /* A_ELEMS_PER_THREAD >= 3 */
#else   /* TRANSA = 0 */
#if (A_ELEMS_PER_THREAD >= 3)
                ll = l + tidLo; 
                IF (ll < llLimit) THEN
                    unsigned int iiLimit = min (i + TILE_DIM, parms.n);
                    offs2 = tidHi;
                    for (ii = i + offs2; ii < iiLimit; ii += COL_INCR) {
                        AA[IDXAA(tidLo, offs2)] = parms.A[IDXA(ll,ii)];
                        offs2 += COL_INCR;
                    }
                ENDIF
#else  /* A_ELEMS_PER_THREAD >= 3 */
#if (A_ELEMS_PER_THREAD >= 1)
                ll = l + tidLo;
                IF (ll < llLimit) THEN
                    ii = i + tidHi;
                    IF (ii < parms.n) THEN
                        unsigned int idxAA;
                        unsigned int addrA;
                        idxAA = IDXAA(tidLo, tidHi);
                        addrA = IDXA(ll, ii);
                        AA[idxAA] = parms.A[addrA];
#if (A_ELEMS_PER_THREAD >= 2)
                        ii += COL_INCR;
                        IF (ii < parms.n) THEN
                            idxAA += COL_INCR * AA_COL_OFS;
                            addrA += COL_INCR * A_COL_OFS;
                            AA[idxAA] = parms.A[addrA];
                        ENDIF
#endif  /* A_ELEMS_PER_THREAD >= 2 */
                    ENDIF
                ENDIF
#endif  /* A_ELEMS_PER_THREAD >= 1 */
#endif  /* A_ELEMS_PER_THREAD >= 3 */
#endif  /* TRANSA = 0 */
                
#if (TRANSB==0)
#if (B_ELEMS_PER_THREAD >= 3)
                ll = l + tidLo;
                IF (ll < llLimit) THEN
                    unsigned int jjLimit = min (j + TILE_DIM, parms.n);
                    offs2 = tidHi;
                    for (jj = j + offs2; jj < jjLimit; jj += COL_INCR) {
                        BB[IDXBB(tidLo, offs2)] = parms.B[IDXB(ll,jj)];
                        offs2 += COL_INCR;
                    }
                ENDIF
#else /* B_ELEMS_PER_THREAD >= 3 */
#if (B_ELEMS_PER_THREAD >= 1)
                ll = l + tidLo;
                IF (ll < llLimit) THEN
                    jj = j + tidHi;
                    IF (jj < parms.n) THEN
                        unsigned int idxBB;
                        unsigned int addrB;
                        idxBB = IDXBB(tidLo,tidHi);
                        addrB = IDXB(ll,jj);
                        BB[idxBB] = parms.B[addrB];
#if (B_ELEMS_PER_THREAD >= 2)
                        jj += COL_INCR;
                        IF (jj < parms.n) THEN
                            idxBB += COL_INCR * BB_COL_OFS;
                            addrB += COL_INCR * B_COL_OFS;
                            BB[idxBB] = parms.B[addrB];
                        ENDIF
#endif  /* B_ELEMS_PER_THREAD >= 2 */
                    ENDIF
                ENDIF
#endif  /* B_ELEMS_PER_THREAD >= 1 */
#endif  /* B_ELEMS_PER_THREAD >= 3 */
#else   /* TRANSB==0 */
#if (B_ELEMS_PER_THREAD >= 3)
                jj = j + tidLo;
                IF (jj < parms.n) THEN
                    offs2 = tidHi;
                    for (ll = l + offs2; ll < llLimit; ll += COL_INCR) {
                        BB[IDXBB(offs2, tidLo)] = parms.B[IDXB(jj,ll)];
                        offs2 += COL_INCR;
                    }
                ENDIF
#else  /* B_ELEMS_PER_THREAD >= 3 */
#if (B_ELEMS_PER_THREAD >= 1)
                jj = j + tidLo;
                IF (jj < parms.n) THEN
                    ll = l + tidHi;
                    IF (ll < llLimit) THEN
                        unsigned int idxBB;
                        unsigned int addrB;
                        idxBB = IDXBB(tidHi,tidLo);
                        addrB = IDXB(jj,ll);
                        BB[idxBB] = parms.B[addrB];
#if (B_ELEMS_PER_THREAD >= 2)
                        ll += COL_INCR;
                        IF (ll < llLimit) THEN
                            idxBB += COL_INCR;
                            addrB += COL_INCR * B_COL_OFS;
                            BB[idxBB] = parms.B[addrB];
                        ENDIF
#endif  /* B_ELEMS_PER_THREAD >= 2 */
                    ENDIF
                ENDIF
#endif  /* B_ELEMS_PER_THREAD >= 1 */
#endif  /* B_ELEMS_PER_THREAD >= 3 */
#endif  /* TRANSB0==0 */
                
                /* Wait until all elements of the A-tile and the B-tile have
                 * been read, before any thread starts with the computation of
                 * dot products
                 */
                __syncthreads ();
                

                /* For each of the result tile elements it needs to compute, a
                 * thread computes the partial dot product by combining the
                 * appropriate row (physically: column, due to transposition of
                 * tile) of A-tile with the appropriate column of the B-tile.
                 * In this case, each thread updates two dot products. Inline
                 * checks prevent computation for result tile elements that do
                 * not correspond to elements inside the result matrix.
                 */
                ii = tidLo;
                IF (ii < (parms.n - i)) THEN
                    unsigned int z = llLimit - l;
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
#error C_ELEMS_PER_THREAD >= 3 not supported
#endif
                    }
                }
            }

            /* At this point each thread has computed the dot product(s) that
             * represent each element of the result matrix tile (i.e. C-tile) 
             * it is responsible for. If beta is zero, don't read the C-tile,
             * otherwise read the C-tile to scale it by beta.
             */
            if (parms.beta == 0.0f) {
                ii = i + tidLo;
                jj = j + tidHi;
                IF ((ii < parms.n) && (jj < parms.n)) THEN
                unsigned int addrC = IDXC(ii,jj);
#if (C_ELEMS_PER_THREAD >= 1)
#if (UPPER==1)
                    if (ii <= jj) {
#else
                    if (ii >= jj) {
#endif
                        parms.C[addrC] = parms.alpha * dp0;
                    }
#if (C_ELEMS_PER_THREAD >= 2)
                    jj += COL_INCR;
                    IF (jj < parms.n) THEN
#if (UPPER==1)
                        if (ii <= jj) {
#else
                        if (ii >= jj) {
#endif
                            addrC += COL_INCR * C_COL_OFS;
                            parms.C[addrC] = parms.alpha * dp1;
                        }
#if (C_ELEMS_PER_THREAD >= 3)
#error C_ELEMS_PER_THREAD >= 3 not supported
#endif /* C_ELEMS_PER_THREAD >= 3 */
                    ENDIF
#endif /* C_ELEMS_PER_THREAD >= 2 */
                ENDIF
#endif /* C_ELEMS_PER_THREAD >= 1 */
            } else {
                ii = i + tidLo;
                jj = j + tidHi;
                IF ((ii < parms.n) && (jj < parms.n)) THEN
                unsigned int addrC = IDXC(ii,jj);
#if (C_ELEMS_PER_THREAD >= 1)
#if (UPPER==1)
                    if (ii <= jj) {
#else
                    if (ii >= jj) {
#endif
                        parms.C[addrC] = parms.beta * parms.C[addrC] + 
                                         parms.alpha * dp0;
                    }
#if (C_ELEMS_PER_THREAD >= 2)
                    jj += COL_INCR;
                    IF (jj < parms.n) THEN
#if (UPPER==1)
                        if (ii <= jj) {
#else
                        if (ii >= jj) {
#endif
                            addrC += COL_INCR * C_COL_OFS;
                            parms.C[addrC] = parms.beta * parms.C[addrC] + 
                                             parms.alpha * dp1;
                        }
#if (C_ELEMS_PER_THREAD >= 3)
#error C_ELEMS_PER_THREAD >= 3 not supported
#endif /* C_ELEMS_PER_THREAD >= 3 */
                    ENDIF
#endif /* C_ELEMS_PER_THREAD >= 2 */
                ENDIF
#endif /* C_ELEMS_PER_THREAD >= 1 */
            }
        }
    }


