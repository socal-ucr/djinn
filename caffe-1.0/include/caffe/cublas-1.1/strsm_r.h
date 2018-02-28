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

    unsigned int i;
    int j;
    unsigned int ii; 
    int jj;
    unsigned int tid;
    unsigned int tidLo;
    unsigned int tidHi;
    unsigned int addr;
#if (ALPHA_IS_ZERO==0)
    unsigned int k; 
    unsigned int kk;
    int x;
    float temp;
    float temp2;
#endif
    tid = threadIdx.x;
    tidLo = tid & (BLK - 1);
    tidHi = tid >> BLK_LOG;

#if (USE_MIXED_STEPPER==1)    
    for (i = IMUL(blockIdx.x,BLK); i < parms.m; i += IINC) {
#else
    {
        i = IMUL(blockIdx.x,BLK);
#endif
#if ((LOWER==1) ^ (TRANS==1))
        for (j = ((parms.n - 1) & (-BLK)); j >= 0; j -= BLK) {
#else
        for (j = 0; j < parms.n; j += BLK) {
#endif
#if (ALPHA_IS_ZERO==1)
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
#else /* ALPHA_IS_ZERO */
            __syncthreads ();

            /* copy block Bij and transpose*/
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
            addr = IDXBB(tidHi,tidLo);
            BB[addr] = temp;
            addr += B_NBR_COLS;
            BB[addr] = temp2;

            /* copy block Ajj */
            ii = j + tidLo;
            jj = j + tidHi;
            temp = 0.0f;
            temp2 = 0.0f;
            IF ((ii < parms.n) && (jj < parms.n)) THEN
                addr = IDXA(ii,jj);
                temp = parms.A[addr];
                jj += A_NBR_COLS;
                IF (jj < parms.n) THEN
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
            /* wait for blocks Bij and Ajj to be loaded */
            __syncthreads ();

            /* solve for Xij, result placed back in Bij */
            if (tid < BLK) {  
                /* FIXME: Any way to get better parallelism? 
                 * Right now we have one thread per column.
                 */
                ii = tid;
#if ((LOWER==1) ^ (TRANS==1))
                x = min ((BLK-1), (parms.n - 1 - j));
                for (jj = x; jj >= 0; jj--) {
#else
                x = min (BLK, parms.n - j);
                for (jj = 0; jj < x; jj++) {
#endif
                    temp = BB[IDXBB(jj,ii)];
#if (NOUNIT==1)
                        temp /= AA[IDXAA(jj,jj)];
#endif
#if ((LOWER==1) ^ (TRANS==1))
                    for (kk = 0; kk < jj; kk++) { 
#else
                    for (kk = (jj + 1); kk < BLK; kk++) {
#endif
                        BB[IDXBB(kk,ii)] -= temp * AA[IDXAA(jj,kk)];
                    }
                    BB[IDXBB(jj,ii)] = temp;
                }
            }
            /* wait for Xij computation to be complete */
            __syncthreads ();
#if ((LOWER==1) ^ (TRANS==1))
            for (k = 0; k < j; k += BLK) {
#else
            for (k = (j + BLK); k < parms.n; k += BLK) {
#endif
                unsigned int tj;
                unsigned int ti;
                /* copy block Ajk */
                __syncthreads ();
#if (TRANS==0)
                jj = j + tidLo;
                kk = k + tidHi;
#else
                jj = k + tidLo;
                kk = j + tidHi;
#endif
                temp = 0.0f;
                temp2 = 0.0f;
                IF ((jj < parms.n) && (kk < parms.n)) THEN
                    addr = IDXA(jj,kk);
                    temp = parms.A[addr];
                    kk += A_NBR_COLS;
                    IF (kk < parms.n) THEN
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
                __syncthreads ();

                /* compute block Bik -= Bij * Ajk */
                ii = tidLo;
                jj = tidHi;
                /* compute dot product */
                temp = 0.0f;
                temp2 = 0.0f;
                tj = IDXAA( 0,jj);
                ti = IDXBB( 0,ii);
                temp += AA[tj +  0] * BB[ti +  0];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS +  0] * BB[ti +  0];
                temp += AA[tj +  1] * BB[ti +  1];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS +  1] * BB[ti +  1];
                temp += AA[tj +  2] * BB[ti +  2];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS +  2] * BB[ti +  2];
                temp += AA[tj +  3] * BB[ti +  3];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS +  3] * BB[ti +  3];
                temp += AA[tj +  4] * BB[ti +  4];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS +  4] * BB[ti +  4];
                temp += AA[tj +  5] * BB[ti +  5];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS +  5] * BB[ti +  5];
                temp += AA[tj +  6] * BB[ti +  6];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS +  6] * BB[ti +  6];
                temp += AA[tj +  7] * BB[ti +  7];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS +  7] * BB[ti +  7];
                temp += AA[tj +  8] * BB[ti +  8];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS +  8] * BB[ti +  8];
                temp += AA[tj +  9] * BB[ti +  9];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS +  9] * BB[ti +  9];
                temp += AA[tj + 10] * BB[ti + 10];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 10] * BB[ti + 10];
                temp += AA[tj + 11] * BB[ti + 11];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 11] * BB[ti + 11];
                temp += AA[tj + 12] * BB[ti + 12];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 12] * BB[ti + 12];
                temp += AA[tj + 13] * BB[ti + 13];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 13] * BB[ti + 13];
                temp += AA[tj + 14] * BB[ti + 14];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 14] * BB[ti + 14];
                temp += AA[tj + 15] * BB[ti + 15];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 15] * BB[ti + 15];
                temp += AA[tj + 16] * BB[ti + 16];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 16] * BB[ti + 16];
                temp += AA[tj + 17] * BB[ti + 17];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 17] * BB[ti + 17];
                temp += AA[tj + 18] * BB[ti + 18];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 18] * BB[ti + 18];
                temp += AA[tj + 19] * BB[ti + 19];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 19] * BB[ti + 19];
                temp += AA[tj + 20] * BB[ti + 20];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 20] * BB[ti + 20];
                temp += AA[tj + 21] * BB[ti + 21];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 21] * BB[ti + 21];
                temp += AA[tj + 22] * BB[ti + 22];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 22] * BB[ti + 22];
                temp += AA[tj + 23] * BB[ti + 23];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 23] * BB[ti + 23];
                temp += AA[tj + 24] * BB[ti + 24];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 24] * BB[ti + 24];
                temp += AA[tj + 25] * BB[ti + 25];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 25] * BB[ti + 25];
                temp += AA[tj + 26] * BB[ti + 26];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 26] * BB[ti + 26];
                temp += AA[tj + 27] * BB[ti + 27];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 27] * BB[ti + 27];
                temp += AA[tj + 28] * BB[ti + 28];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 28] * BB[ti + 28];
                temp += AA[tj + 29] * BB[ti + 29];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 29] * BB[ti + 29];
                temp += AA[tj + 30] * BB[ti + 30];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 30] * BB[ti + 30];
                temp += AA[tj + 31] * BB[ti + 31];
                temp2+= AA[tj+B_NBR_COLS*BB_COL_OFS + 31] * BB[ti + 31];

                IF (((k+jj) < parms.n) && ((i+ii) < parms.m)) THEN
                    addr = IDXB(i+ii,k+jj);
                    parms.B[addr] -= temp;
                    jj += B_NBR_COLS;
                    IF ((k+jj) < parms.n) THEN
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
                parms.B[addr] = parms.alpha * BB[IDXBB(tidHi,tidLo)];
                jj += B_NBR_COLS;
                IF (jj < parms.n) THEN
                    addr += B_NBR_COLS * B_COL_OFS;
                    parms.B[addr] = parms.alpha * BB[IDXBB(tidHi+B_NBR_COLS,tidLo)];
                ENDIF
            ENDIF
#endif /* ALPHA_IS_ZERO */
        }
    }
