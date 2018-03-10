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

#if (A_ELEMS_PER_THREAD!=2)
#error code hardwired for A_ELEMS_PER_THREAD==2
#endif
#if (B_ELEMS_PER_THREAD!=2)
#error code hardwired for B_ELEMS_PER_THREAD==2
#endif

#if (FAST_IMUL==1)
#undef IMUL
#define IMUL(x,y)       __umul24(x,y)
#else
#undef IMUL
#define IMUL(x,y)       ((x)*(y))
#endif

#define IDXA(row,col)   (IMUL(parms.lda,col)+(row))
#define IDXB(row,col)   (IMUL(parms.ldb,col)+(row))
#define IDXAA(row,col)  (__umul24(BLK+1,col)+(row))
#define IDXBB(row,col)  (__umul24(BLK+1,col)+(row))
#define AA_COL_OFS      (IDXAA(0,1)-IDXAA(0,0))
#define BB_COL_OFS      (IDXBB(0,1)-IDXBB(0,0))
#define A_COL_OFS       (IDXA(0,1)-IDXA(0,0))
#define B_COL_OFS       (IDXB(0,1)-IDXB(0,0))
#define C_COL_OFS       (IDXC(0,1)-IDXC(0,0))

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

    int i; 
    unsigned int j; 
    int ii;
    unsigned int jj; 
    unsigned int tid;
    unsigned int tidLo;
    unsigned int tidHi;
#if (ALPHA_IS_ZERO==0)
    unsigned int k;
    unsigned int kk;
    unsigned int addr;
    int x;
    float temp;
    float temp2;
#endif
    tid = threadIdx.x;
    tidLo = (tid & (BLK - 1));
    tidHi = (tid >> BLK_LOG);

#if (USE_MIXED_STEPPER==1)    
    for (j = IMUL(blockIdx.x,BLK); j < parms.n; j += JINC) {
#else
    {
        j = IMUL(blockIdx.x,BLK);
#endif
#if ((LOWER==1) ^ (TRANS==1))
        for (i = 0; i < parms.m; i += BLK) {
#else
        for (i = ((parms.m - 1) & (-BLK)); i >= 0; i -= BLK) {    
#endif
#if (ALPHA_IS_ZERO==1)
            /* set block Bij zero */
            ii = i + tidLo;
            jj = j + tidHi;
            IF ((ii < parms.m) && (jj < parms.n)) THEN
                parms.B[IDXB(ii,jj)] = 0.0f;
                IF ((jj+B_NBR_COLS) < parms.n) THEN
                    parms.B[IDXB(ii,jj+B_NBR_COLS)] = 0.0f;
                ENDIF
            ENDIF
#else /* ALPHA_IS_ZERO */
            __syncthreads ();
            /* copy block Bij */
            ii = i + tidLo;
            jj = j + tidHi;
            temp = 0.0f;
            temp2 = 0.0f;
            IF ((ii < parms.m) && (jj < parms.n)) THEN
                addr = IDXB(ii,jj);
                temp = parms.B[addr];
                jj += B_NBR_COLS;
                IF (jj < parms.n) THEN
                    addr += B_NBR_COLS * B_COL_OFS;
                    temp2 = parms.B[addr]; 
                ENDIF
            ENDIF
            addr = IDXBB(tidLo,tidHi);
            BB[addr] = temp;
            addr += B_NBR_COLS * BB_COL_OFS;
            BB[addr] = temp2;

            /* copy block Aii */
            ii = i + tidLo;
            jj = i + tidHi;
            temp = 0.0f;
            temp2 = 0.0f;
            IF ((ii < parms.m) && (jj < parms.m)) THEN
                addr = IDXA(ii,jj);
                temp = parms.A[addr];
                jj += A_NBR_COLS;
                IF (jj < parms.m) THEN
                    addr += A_NBR_COLS * A_COL_OFS;
                    temp2 = parms.A[addr];
                ENDIF
            ENDIF
#if (TRANS==0)
            addr = IDXAA(tidLo,tidHi);
            AA[addr] = temp;
            addr += A_NBR_COLS * AA_COL_OFS;
            AA[addr] = temp2;
#else
            addr = IDXAA(tidHi,tidLo);
            AA[addr] = temp;
            addr += A_NBR_COLS;
            AA[addr] = temp2;
#endif

            /* wait for blocks Bij and Aii to be loaded */
            __syncthreads ();

            /* solve for Xij, result placed back in Bij */
            if (tid < BLK) {  
                /* FIXME: Any way to get better parallelism? 
                 * Right now we have one thread per column.
                 */
                jj = tid;
#if ((LOWER==1) ^ (TRANS==1))
                x = min (BLK, parms.m - i);
                for (ii = 0; ii < x; ii++) {
#else
                x = min ((BLK-1), (parms.m-1-i));
                for (ii = x; ii >= 0; ii--) {
#endif
                    temp = BB[IDXBB(ii,jj)];
#if (NOUNIT==1)
                        temp /= AA[IDXAA(ii,ii)];
#endif
#if ((LOWER==1) ^ (TRANS==1))
                    for (kk = (ii + 1); kk < BLK; kk++) {
#else
                    for (kk = 0; kk < ii; kk++) {
#endif
                        BB[IDXBB(kk,jj)] -= temp * AA[IDXAA(kk,ii)];
                    }
                    BB[IDXBB(ii,jj)] = temp;
                }
            }
            /* wait for Xij computation to be complete */
            __syncthreads ();
#if ((LOWER==1) ^ (TRANS==1))
            for (k = (i + BLK); k < parms.m; k += BLK) {
#else
            for (k = 0; k < i; k += BLK) {
#endif
                unsigned int ti;
                unsigned int tj;

                /* copy block Aki */
                __syncthreads ();
#if (TRANS==0)
                kk = k + tidLo;
                ii = i + tidHi;
                temp = 0.0f;
                temp2 = 0.0f;
                IF ((kk < parms.m) && (ii < parms.m)) THEN
                    addr = IDXA(kk,ii);
                    temp = parms.A[addr];
                    ii += A_NBR_COLS;
                    IF (ii < parms.m) THEN
                        addr += A_NBR_COLS * A_COL_OFS;
                        temp2 = parms.A[addr];
                    ENDIF
                ENDIF
                addr = IDXAA(tidHi,tidLo);
                AA[addr] = temp;
                addr += A_NBR_COLS;
                AA[addr] = temp2;
#else
                ii = i + tidLo;
                kk = k + tidHi;
                temp = 0.0f;
                temp2 = 0.0f;
                IF ((kk < parms.m) && (ii < parms.m)) THEN
                    addr = IDXA(ii,kk);
                    temp = parms.A[addr];
                    kk += A_NBR_COLS;
                    IF (kk < parms.m) THEN
                        addr += A_NBR_COLS * A_COL_OFS;
                        temp2 = parms.A[addr];
                    ENDIF
                ENDIF
                addr = IDXAA(tidLo,tidHi);
                AA[addr] = temp;
                addr += A_NBR_COLS * AA_COL_OFS;
                AA[addr] = temp2;
#endif
                __syncthreads ();

                /* compute block Bkj -= Bij * Aki */
                ii = tidLo;
                jj = tidHi;

                /* compute dot products */
                temp  = 0.0f;
                temp2 = 0.0f;
                ti = IDXAA( 0,ii);
                tj = IDXBB( 0,jj);
                temp += AA[ti +  0] * BB[tj +  0];
                temp2+= AA[ti +  0] * BB[tj +  0+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti +  1] * BB[tj +  1];
                temp2+= AA[ti +  1] * BB[tj +  1+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti +  2] * BB[tj +  2];
                temp2+= AA[ti +  2] * BB[tj +  2+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti +  3] * BB[tj +  3];
                temp2+= AA[ti +  3] * BB[tj +  3+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti +  4] * BB[tj +  4];
                temp2+= AA[ti +  4] * BB[tj +  4+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti +  5] * BB[tj +  5];
                temp2+= AA[ti +  5] * BB[tj +  5+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti +  6] * BB[tj +  6];
                temp2+= AA[ti +  6] * BB[tj +  6+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti +  7] * BB[tj +  7];
                temp2+= AA[ti +  7] * BB[tj +  7+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti +  8] * BB[tj +  8];
                temp2+= AA[ti +  8] * BB[tj +  8+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti +  9] * BB[tj +  9];
                temp2+= AA[ti +  9] * BB[tj +  9+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 10] * BB[tj + 10];
                temp2+= AA[ti + 10] * BB[tj + 10+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 11] * BB[tj + 11];
                temp2+= AA[ti + 11] * BB[tj + 11+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 12] * BB[tj + 12];
                temp2+= AA[ti + 12] * BB[tj + 12+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 13] * BB[tj + 13];
                temp2+= AA[ti + 13] * BB[tj + 13+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 14] * BB[tj + 14];
                temp2+= AA[ti + 14] * BB[tj + 14+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 15] * BB[tj + 15];
                temp2+= AA[ti + 15] * BB[tj + 15+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 16] * BB[tj + 16];
                temp2+= AA[ti + 16] * BB[tj + 16+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 17] * BB[tj + 17];
                temp2+= AA[ti + 17] * BB[tj + 17+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 18] * BB[tj + 18];
                temp2+= AA[ti + 18] * BB[tj + 18+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 19] * BB[tj + 19];
                temp2+= AA[ti + 19] * BB[tj + 19+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 20] * BB[tj + 20];
                temp2+= AA[ti + 20] * BB[tj + 20+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 21] * BB[tj + 21];
                temp2+= AA[ti + 21] * BB[tj + 21+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 22] * BB[tj + 22];
                temp2+= AA[ti + 22] * BB[tj + 22+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 23] * BB[tj + 23];
                temp2+= AA[ti + 23] * BB[tj + 23+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 24] * BB[tj + 24];
                temp2+= AA[ti + 24] * BB[tj + 24+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 25] * BB[tj + 25];
                temp2+= AA[ti + 25] * BB[tj + 25+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 26] * BB[tj + 26];
                temp2+= AA[ti + 26] * BB[tj + 26+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 27] * BB[tj + 27];
                temp2+= AA[ti + 27] * BB[tj + 27+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 28] * BB[tj + 28];
                temp2+= AA[ti + 28] * BB[tj + 28+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 29] * BB[tj + 29];
                temp2+= AA[ti + 29] * BB[tj + 29+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 30] * BB[tj + 30];
                temp2+= AA[ti + 30] * BB[tj + 30+B_NBR_COLS*BB_COL_OFS];
                temp += AA[ti + 31] * BB[tj + 31];
                temp2+= AA[ti + 31] * BB[tj + 31+B_NBR_COLS*BB_COL_OFS];

                IF (((k+ii) < parms.m) && ((j+jj) < parms.n)) THEN
                    addr = IDXB(k+ii,j+jj);
                    parms.B[addr] -= temp;
                    jj += B_NBR_COLS;
                    IF ((j+jj) < parms.n) THEN
                        addr += B_NBR_COLS * B_COL_OFS;
                        parms.B[addr] -= temp2;
                    ENDIF
                ENDIF
            }

            __syncthreads ();
            /* write back block alpha * Bij */
            ii = i + tidLo;
            jj = j + tidHi;
            IF ((ii < parms.m) && (jj < parms.n)) THEN
                addr = IDXB(ii,jj);
                parms.B[addr] = parms.alpha * BB[IDXBB(tidLo,tidHi)];
                jj += B_NBR_COLS;
                IF (jj < parms.n) THEN
                    addr += B_NBR_COLS * B_COL_OFS;
                    parms.B[addr] = parms.alpha * BB[IDXBB(tidLo,tidHi+B_NBR_COLS)];
                ENDIF
            ENDIF
#endif /* ALPHA_IS_ZERO */
        }
    }
