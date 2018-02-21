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

    int i, j, ii, jj;
    unsigned int addr;
#if (ALPHA0==0)
    int k, kk;
    unsigned int ti;
    unsigned int tj;
    float temp, temp2;
    float dot, dot2;
#endif
    unsigned int tid = threadIdx.x;
    unsigned int tidLo = (tid & (BLK - 1));
    unsigned int tidHi = (tid >> BLK_LOG);
#if (USE_MIXED_STEPPER==1)
    for (i = IMUL(blockIdx.x,BLK); i < parms.m; i += IINC) {
#else
    {   
        i = IMUL(blockIdx.x,BLK);
#endif
#if ((LOWER==1) ^ (TRANS==1))
        for (j = 0; j < parms.n; j += BLK) {
#else
        for (j = ((parms.n - 1) & (-BLK)); j >= 0; j -= BLK) {
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
            ii = tidLo;
            jj = tidHi;
            dot = 0.0f;
            dot2 = 0.0f;

#if ((LOWER==1) ^ (TRANS==1))
            for (k = j; k < parms.n; k += BLK) {
#else
            for (k = j; k >= 0; k -= BLK) {       
#endif
                __syncthreads ();
                /* copy block Bik */
                ii = i + tidLo;
                kk = k + tidHi;
                temp = 0.0f;
                temp2 = 0.0f;
                IF ((ii < parms.m) && (kk < parms.n)) THEN
                    addr = IDXB(ii,kk);
                    temp = parms.B[addr];
                    kk += B_NBR_COLS;
                    IF (kk < parms.n) THEN
                        addr += B_NBR_COLS * B_COL_OFS;
                        temp2 = parms.B[addr];
                    ENDIF
                ENDIF
                addr = IDXBB(tidHi,tidLo);    
                BB[addr] = temp;
                addr += B_NBR_COLS;
                BB[addr] = temp2;
                /* copy block Akj */
#if (TRANS==0)
                kk = k + tidLo;
                jj = j + tidHi;
                temp = 0.0f;
                temp2 = 0.0f;
                addr = IDXA(kk,jj);
                IF ((kk < parms.n) && (jj < parms.n)) THEN
#if (LOWER==1)
                    if (jj <= kk) {
#else
                    if (jj >= kk) {
#endif
#if (UNIT==1)
                        temp = (kk == jj) ? 1.0f : parms.A[addr];
#else
                        temp = parms.A[addr];
#endif
                    }
                    jj += A_NBR_COLS;
                    addr += A_NBR_COLS * A_COL_OFS;
                    IF (jj < parms.n) THEN
#if (LOWER==1)
                        if (jj <= kk) {
#else
                        if (jj >= kk) {
#endif
#if (UNIT==1)
                            temp2 = (kk == jj) ? 1.0f : parms.A[addr];
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

#else /* TRANS ----------------------------------------------*/
                jj = j + tidLo;
                kk = k + tidHi;
                temp = 0.0f;
                temp2 = 0.0f;
                addr = IDXA(jj,kk);
                IF ((kk < parms.n) && (jj < parms.n)) THEN
#if (LOWER==0)
                    if (jj <= kk) {
#else
                    if (jj >= kk) {
#endif
#if (UNIT==1)
                        temp = (kk == jj) ? 1.0f : parms.A[addr];
#else /* UNIT */
                        temp = parms.A[addr];
#endif
                    }
                    kk += A_NBR_COLS;
                    addr += A_NBR_COLS * A_COL_OFS;
                    IF (kk < parms.n) THEN
#if (LOWER==0)
                        if (jj <= kk) {
#else
                        if (jj >= kk) {
#endif
#if (UNIT==1)
                            temp2 = (kk == jj) ? 1.0f : parms.A[addr];
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
#endif /* TRANS */
                __syncthreads ();
                /* bb += Bik * Akj */
                ii = tidLo;
                jj = tidHi;
                /* compute dot product */
                ti = IDXBB( 0,ii);
                tj = IDXAA( 0,jj);
                dot += AA[tj +  0] * BB[ti +  0];
                dot2+= AA[tj +  0 + A_NBR_COLS*AA_COL_OFS] * BB[ti +  0];
                dot += AA[tj +  1] * BB[ti +  1];
                dot2+= AA[tj +  1 + A_NBR_COLS*AA_COL_OFS] * BB[ti +  1];
                dot += AA[tj +  2] * BB[ti +  2];
                dot2+= AA[tj +  2 + A_NBR_COLS*AA_COL_OFS] * BB[ti +  2];
                dot += AA[tj +  3] * BB[ti +  3];
                dot2+= AA[tj +  3 + A_NBR_COLS*AA_COL_OFS] * BB[ti +  3];
                dot += AA[tj +  4] * BB[ti +  4];
                dot2+= AA[tj +  4 + A_NBR_COLS*AA_COL_OFS] * BB[ti +  4];
                dot += AA[tj +  5] * BB[ti +  5];
                dot2+= AA[tj +  5 + A_NBR_COLS*AA_COL_OFS] * BB[ti +  5];
                dot += AA[tj +  6] * BB[ti +  6];
                dot2+= AA[tj +  6 + A_NBR_COLS*AA_COL_OFS] * BB[ti +  6];
                dot += AA[tj +  7] * BB[ti +  7];
                dot2+= AA[tj +  7 + A_NBR_COLS*AA_COL_OFS] * BB[ti +  7];
                dot += AA[tj +  8] * BB[ti +  8];
                dot2+= AA[tj +  8 + A_NBR_COLS*AA_COL_OFS] * BB[ti +  8];
                dot += AA[tj +  9] * BB[ti +  9];
                dot2+= AA[tj +  9 + A_NBR_COLS*AA_COL_OFS] * BB[ti +  9];
                dot += AA[tj + 10] * BB[ti + 10];
                dot2+= AA[tj + 10 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 10];
                dot += AA[tj + 11] * BB[ti + 11];
                dot2+= AA[tj + 11 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 11];
                dot += AA[tj + 12] * BB[ti + 12];
                dot2+= AA[tj + 12 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 12];
                dot += AA[tj + 13] * BB[ti + 13];
                dot2+= AA[tj + 13 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 13];
                dot += AA[tj + 14] * BB[ti + 14];
                dot2+= AA[tj + 14 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 14];
                dot += AA[tj + 15] * BB[ti + 15];
                dot2+= AA[tj + 15 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 15];
                dot += AA[tj + 16] * BB[ti + 16];
                dot2+= AA[tj + 16 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 16];
                dot += AA[tj + 17] * BB[ti + 17];
                dot2+= AA[tj + 17 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 17];
                dot += AA[tj + 18] * BB[ti + 18];
                dot2+= AA[tj + 18 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 18];
                dot += AA[tj + 19] * BB[ti + 19];
                dot2+= AA[tj + 19 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 19];
                dot += AA[tj + 20] * BB[ti + 20];
                dot2+= AA[tj + 20 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 20];
                dot += AA[tj + 21] * BB[ti + 21];
                dot2+= AA[tj + 21 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 21];
                dot += AA[tj + 22] * BB[ti + 22];
                dot2+= AA[tj + 22 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 22];
                dot += AA[tj + 23] * BB[ti + 23];
                dot2+= AA[tj + 23 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 23];
                dot += AA[tj + 24] * BB[ti + 24];
                dot2+= AA[tj + 24 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 24];
                dot += AA[tj + 25] * BB[ti + 25];
                dot2+= AA[tj + 25 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 25];
                dot += AA[tj + 26] * BB[ti + 26];
                dot2+= AA[tj + 26 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 26];
                dot += AA[tj + 27] * BB[ti + 27];
                dot2+= AA[tj + 27 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 27];
                dot += AA[tj + 28] * BB[ti + 28];
                dot2+= AA[tj + 28 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 28];
                dot += AA[tj + 29] * BB[ti + 29];
                dot2+= AA[tj + 29 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 29];
                dot += AA[tj + 30] * BB[ti + 30];
                dot2+= AA[tj + 30 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 30];
                dot += AA[tj + 31] * BB[ti + 31];
                dot2+= AA[tj + 31 + A_NBR_COLS*AA_COL_OFS] * BB[ti + 31];
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

