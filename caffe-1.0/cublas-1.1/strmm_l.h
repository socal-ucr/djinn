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

#define IDXA(row,col)   (IMUL(parms.lda,(col))+(row))
#define IDXB(row,col)   (IMUL(parms.ldb,(col))+(row))
#define IDXAA(row,col)  (__umul24((BLK+1),(col))+(row))
#define IDXBB(row,col)  (__umul24((BLK+1),(col))+(row))

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

    int i, ii;
    unsigned int j, jj;
    unsigned int addr;

    unsigned tid = threadIdx.x;
    unsigned int tidLo = (tid & (BLK - 1));
    unsigned int tidHi = (tid >> BLK_LOG);
#if (ALPHA0==0)
    int k, kk;
    unsigned int ti;
    unsigned int tj;
    float dot;
    float dot2;
    float temp;
    float temp2;
#endif

#if ((LOWER==0) ^ (TRANS==1))
    for (i = 0; i < parms.m; i += BLK) {
#else
    for (i = ((parms.m - 1) & (-BLK)); i >= 0; i -= BLK) {  
#endif
#if (USE_MIXED_STEPPER==1)
        for (j = IMUL(blockIdx.x,BLK); j < parms.n; j += JINC) {
#else
        {
            j = IMUL(blockIdx.x,BLK);
#endif
#if (ALPHA0==1)
            /* set block Bij zero */
            ii = i + tidLo;
            jj = j + tidHi;
            IF ((ii < parms.m) && (jj < parms.n)) THEN
                addr = IDXB(ii,jj);
                parms.B[addr] = 0.0f;
                jj += B_NBR_COLS;
                IF (jj < parms.n) THEN
                    addr += B_NBR_COLS * B_COL_OFS;
                    parms.B[addr] = 0.0f;
                ENDIF
            ENDIF
#else
            /* set bb to zero */
            dot  = 0.0f;
            dot2 = 0.0f;
                
#if ((LOWER==0) ^ (TRANS==1))
            for (k = i; k < parms.m; k += BLK) {
#else
            for (k = i; k >= 0; k -= BLK) {    
#endif
                __syncthreads ();
                /* copy block Bkj */
                kk = k + tidLo;
                jj = j + tidHi;
                temp = 0.0f;
                temp2 = 0.0f;
                IF ((kk < parms.m) && (jj < parms.n)) THEN
                    addr = IDXB(kk,jj);
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
                /* copy block Aik */
#if (TRANS==0)
                kk = k + tidHi;
                ii = i + tidLo;
                temp = 0.0f;
                temp2 = 0.0f;
                addr = IDXA(ii,kk);
                IF ((ii < parms.m) && (kk < parms.m)) THEN
#if (LOWER==0)
                    if (ii <= kk) {
#else
                    if (ii >= kk) {
#endif
#if (UNIT==1)
                        temp = (ii == kk) ? 1.0f : parms.A[addr];
#else
                        temp = parms.A[addr];
#endif
                    }
                    kk += A_NBR_COLS;
                    addr += A_NBR_COLS * A_COL_OFS;
                    IF (kk < parms.m) THEN
#if (LOWER==0)
                        if (ii <= kk) {
#else
                        if (ii >= kk) {
#endif
#if (UNIT==1)
                            temp2 = (ii == kk) ? 1.0f : parms.A[addr];
#else
                            temp2 = parms.A[addr];
#endif
                        }
                    ENDIF
                ENDIF
                addr = IDXAA(tidHi,tidLo);
                AA[addr] = temp;
                addr += A_NBR_COLS;
                AA[addr] = temp2;

#else  /*******************************************************************/
                kk = k + tidLo;
                ii = i + tidHi;
                temp = 0.0f;
                temp2 = 0.0f;
                addr = IDXA(kk,ii);
                IF ((ii < parms.m) && (kk < parms.m)) THEN
#if (LOWER==1)
                    if (ii <= kk) {
#else
                    if (ii >= kk) {
#endif
#if (UNIT==1)
                        temp = (ii == kk) ? 1.0f : parms.A[addr];
#else
                        temp = parms.A[addr];
#endif
                    }
                    ii += A_NBR_COLS;
                    addr += A_NBR_COLS * A_COL_OFS;                     
                    IF (ii < parms.m) THEN
#if (LOWER==1)
                        if (ii <= kk) {
#else
                        if (ii >= kk) {
#endif
#if (UNIT==1)
                            temp2 = (ii == kk) ? 1.0f : parms.A[addr];
#else
                            temp2 = parms.A[addr];
#endif
                        }
                    ENDIF
                ENDIF
                addr = IDXAA(tidLo,tidHi);
                AA[addr] = temp;
                addr += A_NBR_COLS * AA_COL_OFS;
                AA[addr] = temp2;
#endif
                __syncthreads ();
                /* bb += Aik * Bkj */
                ii = tidLo;
                jj = tidHi;
                /* compute dot product */
                ti = IDXAA( 0,ii);
                tj = IDXBB( 0,jj);
                dot += AA[ti +  0] * BB[tj +  0];
                dot2+= AA[ti +  0] * BB[tj +  0 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti +  1] * BB[tj +  1];
                dot2+= AA[ti +  1] * BB[tj +  1 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti +  2] * BB[tj +  2];
                dot2+= AA[ti +  2] * BB[tj +  2 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti +  3] * BB[tj +  3];
                dot2+= AA[ti +  3] * BB[tj +  3 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti +  4] * BB[tj +  4];
                dot2+= AA[ti +  4] * BB[tj +  4 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti +  5] * BB[tj +  5];
                dot2+= AA[ti +  5] * BB[tj +  5 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti +  6] * BB[tj +  6];
                dot2+= AA[ti +  6] * BB[tj +  6 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti +  7] * BB[tj +  7];
                dot2+= AA[ti +  7] * BB[tj +  7 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti +  8] * BB[tj +  8];
                dot2+= AA[ti +  8] * BB[tj +  8 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti +  9] * BB[tj +  9];
                dot2+= AA[ti +  9] * BB[tj +  9 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 10] * BB[tj + 10];
                dot2+= AA[ti + 10] * BB[tj + 10 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 11] * BB[tj + 11];
                dot2+= AA[ti + 11] * BB[tj + 11 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 12] * BB[tj + 12];
                dot2+= AA[ti + 12] * BB[tj + 12 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 13] * BB[tj + 13];
                dot2+= AA[ti + 13] * BB[tj + 13 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 14] * BB[tj + 14];
                dot2+= AA[ti + 14] * BB[tj + 14 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 15] * BB[tj + 15];
                dot2+= AA[ti + 15] * BB[tj + 15 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 16] * BB[tj + 16];
                dot2+= AA[ti + 16] * BB[tj + 16 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 17] * BB[tj + 17];
                dot2+= AA[ti + 17] * BB[tj + 17 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 18] * BB[tj + 18];
                dot2+= AA[ti + 18] * BB[tj + 18 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 19] * BB[tj + 19];
                dot2+= AA[ti + 19] * BB[tj + 19 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 20] * BB[tj + 20];
                dot2+= AA[ti + 20] * BB[tj + 20 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 21] * BB[tj + 21];
                dot2+= AA[ti + 21] * BB[tj + 21 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 22] * BB[tj + 22];
                dot2+= AA[ti + 22] * BB[tj + 22 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 23] * BB[tj + 23];
                dot2+= AA[ti + 23] * BB[tj + 23 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 24] * BB[tj + 24];
                dot2+= AA[ti + 24] * BB[tj + 24 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 25] * BB[tj + 25];
                dot2+= AA[ti + 25] * BB[tj + 25 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 26] * BB[tj + 26];
                dot2+= AA[ti + 26] * BB[tj + 26 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 27] * BB[tj + 27];
                dot2+= AA[ti + 27] * BB[tj + 27 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 28] * BB[tj + 28];
                dot2+= AA[ti + 28] * BB[tj + 28 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 29] * BB[tj + 29];
                dot2+= AA[ti + 29] * BB[tj + 29 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 30] * BB[tj + 30];
                dot2+= AA[ti + 30] * BB[tj + 30 + B_NBR_COLS*BB_COL_OFS];
                dot += AA[ti + 31] * BB[tj + 31];
                dot2+= AA[ti + 31] * BB[tj + 31 + B_NBR_COLS*BB_COL_OFS];
            }
            __syncthreads ();
            /* write back Bij = alpha * bb */
            ii = i + tidLo;
            jj = j + tidHi;
            IF ((ii < parms.m) && (jj < parms.n)) THEN
                addr = IDXB(ii,jj);
                parms.B[addr] = parms.alpha * dot;
                jj += B_NBR_COLS;
                IF (jj < parms.n) THEN
                    addr += B_NBR_COLS * B_COL_OFS;
                    parms.B[addr] = parms.alpha * dot2;
                ENDIF
            ENDIF
#endif /* ALPHA0 */
        }
    }

