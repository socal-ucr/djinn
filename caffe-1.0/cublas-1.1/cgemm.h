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

#define IDXA(row,col)   (IMUL(parms.lda,col)+(row))   /* index into matrix A */
#define IDXB(row,col)   (IMUL(parms.ldb,col)+(row))   /* index into matrix B */
#define IDXC(row,col)   (IMUL(parms.ldc,col)+(row))   /* index into matrix C */
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

#define ACCUMULATE_DOT_PRODUCT_16() \
do {                                \
    float s;                        \
    s =         AA_r[li+ 0];        \
    dp.x += s * BB_r[lj+ 0];        \
    dp.y += s * BB_i[lj+ 0];        \
    s =         AA_i[li+ 0];        \
    dp.x -= s * BB_i[lj+ 0];        \
    dp.y += s * BB_r[lj+ 0];        \
    s =         AA_r[li+ 1];        \
    dp.x += s * BB_r[lj+ 1];        \
    dp.y += s * BB_i[lj+ 1];        \
    s =         AA_i[li+ 1];        \
    dp.x -= s * BB_i[lj+ 1];        \
    dp.y += s * BB_r[lj+ 1];        \
    s =         AA_r[li+ 2];        \
    dp.x += s * BB_r[lj+ 2];        \
    dp.y += s * BB_i[lj+ 2];        \
    s =         AA_i[li+ 2];        \
    dp.x -= s * BB_i[lj+ 2];        \
    dp.y += s * BB_r[lj+ 2];        \
    s =         AA_r[li+ 3];        \
    dp.x += s * BB_r[lj+ 3];        \
    dp.y += s * BB_i[lj+ 3];        \
    s =         AA_i[li+ 3];        \
    dp.x -= s * BB_i[lj+ 3];        \
    dp.y += s * BB_r[lj+ 3];        \
    s =         AA_r[li+ 4];        \
    dp.x += s * BB_r[lj+ 4];        \
    dp.y += s * BB_i[lj+ 4];        \
    s =         AA_i[li+ 4];        \
    dp.x -= s * BB_i[lj+ 4];        \
    dp.y += s * BB_r[lj+ 4];        \
    s =         AA_r[li+ 5];        \
    dp.x += s * BB_r[lj+ 5];        \
    dp.y += s * BB_i[lj+ 5];        \
    s =         AA_i[li+ 5];        \
    dp.x -= s * BB_i[lj+ 5];        \
    dp.y += s * BB_r[lj+ 5];        \
    s =         AA_r[li+ 6];        \
    dp.x += s * BB_r[lj+ 6];        \
    dp.y += s * BB_i[lj+ 6];        \
    s =         AA_i[li+ 6];        \
    dp.x -= s * BB_i[lj+ 6];        \
    dp.y += s * BB_r[lj+ 6];        \
    s =         AA_r[li+ 7];        \
    dp.x += s * BB_r[lj+ 7];        \
    dp.y += s * BB_i[lj+ 7];        \
    s =         AA_i[li+ 7];        \
    dp.x -= s * BB_i[lj+ 7];        \
    dp.y += s * BB_r[lj+ 7];        \
    s =         AA_r[li+ 8];        \
    dp.x += s * BB_r[lj+ 8];        \
    dp.y += s * BB_i[lj+ 8];        \
    s =         AA_i[li+ 8];        \
    dp.x -= s * BB_i[lj+ 8];        \
    dp.y += s * BB_r[lj+ 8];        \
    s =         AA_r[li+ 9];        \
    dp.x += s * BB_r[lj+ 9];        \
    dp.y += s * BB_i[lj+ 9];        \
    s =         AA_i[li+ 9];        \
    dp.x -= s * BB_i[lj+ 9];        \
    dp.y += s * BB_r[lj+ 9];        \
    s =         AA_r[li+10];        \
    dp.x += s * BB_r[lj+10];        \
    dp.y += s * BB_i[lj+10];        \
    s =         AA_i[li+10];        \
    dp.x -= s * BB_i[lj+10];        \
    dp.y += s * BB_r[lj+10];        \
    s =         AA_r[li+11];        \
    dp.x += s * BB_r[lj+11];        \
    dp.y += s * BB_i[lj+11];        \
    s =         AA_i[li+11];        \
    dp.x -= s * BB_i[lj+11];        \
    dp.y += s * BB_r[lj+11];        \
    s =         AA_r[li+12];        \
    dp.x += s * BB_r[lj+12];        \
    dp.y += s * BB_i[lj+12];        \
    s =         AA_i[li+12];        \
    dp.x -= s * BB_i[lj+12];        \
    dp.y += s * BB_r[lj+12];        \
    s =         AA_r[li+13];        \
    dp.x += s * BB_r[lj+13];        \
    dp.y += s * BB_i[lj+13];        \
    s =         AA_i[li+13];        \
    dp.x -= s * BB_i[lj+13];        \
    dp.y += s * BB_r[lj+13];        \
    s =         AA_r[li+14];        \
    dp.x += s * BB_r[lj+14];        \
    dp.y += s * BB_i[lj+14];        \
    s =         AA_i[li+14];        \
    dp.x -= s * BB_i[lj+14];        \
    dp.y += s * BB_r[lj+14];        \
    s =         AA_r[li+15];        \
    dp.x += s * BB_r[lj+15];        \
    dp.y += s * BB_i[lj+15];        \
    s =         AA_i[li+15];        \
    dp.x -= s * BB_i[lj+15];        \
    dp.y += s * BB_r[lj+15];        \
} while (0)

#define ACCUMULATE_DOT_PRODUCT_N() \
do {                               \
    float s;                       \
    do {                           \
        s =         AA_r[li+ 0];   \
        dp.x += s * BB_r[lj+ 0];   \
        dp.y += s * BB_i[lj+ 0];   \
        s =         AA_i[li+ 0];   \
        dp.x -= s * BB_i[lj+ 0];   \
        dp.y += s * BB_r[lj+ 0];   \
        li++;                      \
        lj++;                      \
        ll--;                      \
    } while(ll);                   \
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

    unsigned int i, j, l, ii, jj, ll, tid;
    
    cuComplex dp;

    tid = threadIdx.x;

    unsigned tidLo = (tid & (TILE_DIM - 1));
    unsigned tidHi = (tid >> TILE_DIM_LOG);
#if (USE_MIXED_STEPPER == 1)
    for (i = IMUL(blockIdx.y, TILE_DIM); i < parms.m; i += SUP_TILE_DIM) {
       for (j = IMUL(blockIdx.x, TILE_DIM); j < parms.n; j += SUP_TILE_DIM) {
#else
    {        
        {
            i = IMUL(blockIdx.y, TILE_DIM);
            j = IMUL(blockIdx.x, TILE_DIM);
#endif

            /* set accumulation to 0*/
            dp = make_cuComplex (0.0f, 0.0f);

            for (l = 0; l < parms.k; l += TILE_DIM) {
                unsigned int llLimit = min ((l + TILE_DIM), parms.k);
                unsigned int z = llLimit - l;
                unsigned int offs1, offs2;
                cuComplex temp;
                
                /* Wait before clobbering old cache contents */
                __syncthreads ();
                
#if (TRANSA==0)
                offs1 = tidLo;
                offs2 = tidHi;
                ii = i + offs1;
                ll = l + offs2;
                IF ((ii < parms.m) && (ll < llLimit)) THEN
                    unsigned int idxAA;
                    unsigned int addrA;
                    idxAA = IDXAA(offs2, offs1);
                    addrA = IDXA(ii,ll);
                    temp = fetchA(addrA);
                    AA_r[idxAA] = cuCrealf (temp);
                    AA_i[idxAA] = cuCimagf (temp);
                ENDIF
#else   /* TRANSA = 0 */
                offs1 = tidLo;
                offs2 = tidHi;
                ll = l + offs1;
                ii = i + offs2;
                IF ((ll < llLimit) && (ii < parms.m)) THEN
                    unsigned int idxAA;
                    unsigned int addrA;
                    idxAA = IDXAA(offs1,offs2);
                    addrA = IDXA(ll,ii);
                    temp = fetchA(addrA);
#if (CONJGA==0)
                    AA_r[idxAA] = cuCrealf (temp);
                    AA_i[idxAA] = cuCimagf (temp);
#else
                    AA_r[idxAA] = cuCrealf (cuConjf (temp));
                    AA_i[idxAA] = cuCimagf (cuConjf (temp));
#endif  /* CONJGA==0 */
                ENDIF
#endif  /* TRANSA = 0 */
#if (TRANSB==0)
                offs1 = tidLo;  
                offs2 = tidHi;
                ll = l + offs1;
                jj = j + offs2;
                IF ((ll < llLimit) && (jj < parms.n)) THEN
                    unsigned int idxBB;
                    unsigned int addrB;
                    idxBB = IDXBB(offs1,offs2);
                    addrB = IDXB(ll,jj);
                    temp = fetchB(addrB);
                    BB_r[idxBB] = cuCrealf (temp);
                    BB_i[idxBB] = cuCimagf (temp);
                ENDIF
#else   /* TRANSB==0 */
                offs1 = tidLo;
                offs2 = tidHi;
                jj = j + offs1;
                ll = l + offs2;
                IF ((jj < parms.n) && (ll < llLimit)) THEN
                    unsigned int idxBB;
                    unsigned int addrB;
                    idxBB = IDXBB(offs2,offs1);
                    addrB = IDXB(jj,ll);
                    temp = fetchB(addrB);
#if (CONJGB==0)
                    BB_r[idxBB] = cuCrealf (temp);
                    BB_i[idxBB] = cuCimagf (temp);
#else
                    BB_r[idxBB] = cuCrealf (cuConjf (temp));
                    BB_i[idxBB] = cuCimagf (cuConjf (temp));
#endif  /* CONJGB==0 */
                ENDIF
#endif  /* TRANSB0==0 */
                
                /* Wait until new cache contents ready */
                __syncthreads ();
                
                /* We don't iterate over jj and ii since this is all done
                 * in parallel by the threads in each CTA.
                 */
                ii = tidLo;
                jj = tidHi;
                IF (ii < (parms.m - i)) THEN
                    IF (z == TILE_DIM) THEN
                        IF (jj < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ACCUMULATE_DOT_PRODUCT_16();
                        ENDIF
                    ELSE                      
                        IF (jj < (parms.n - j)) THEN
                            unsigned int li = IDXAA(0,ii);
                            unsigned int lj = IDXBB(0,jj);
                            ll = z;
                            ACCUMULATE_DOT_PRODUCT_N();
                        ENDIF
                    ENDIF
                ENDIF
            }
            /* write out completed tile of matrix C */
            if ((parms.beta.x == 0.0f) && (parms.beta.y == 0.0f)) {
                ii = i + tidLo;
                jj = j + tidHi;
                IF ((ii < parms.m) && (jj < parms.n)) THEN
                    unsigned int addrC = IDXC(ii,jj);
                    parms.C[addrC] = cuCmulf (parms.alpha, dp);
                ENDIF
            } else {
                ii = i + tidLo;
                jj = j + tidHi;
                IF ((ii < parms.m) && (jj < parms.n)) THEN
                    unsigned int addrC = IDXC(ii,jj);
                    parms.C[addrC] = cuCaddf(cuCmulf(parms.beta, parms.C[addrC]),
                                     cuCmulf(parms.alpha, dp));
                ENDIF
            }
        }
    }


