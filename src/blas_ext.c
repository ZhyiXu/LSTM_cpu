//
//  blas_ext.c
//  rnn_test
//
//  Created by xuzhuoyi on 2016/12/30.
//  Copyright © 2016年 xuzhuoyi. All rights reserved.
//

#include "blas_ext.h"

void gemm(
          const enum CBLAS_TRANSPOSE TransA,
          const enum CBLAS_TRANSPOSE TransB,
          const int M,
          const int N,
          const int K,
          const DataType alpha,
          const DataType *A,
          const DataType *B,
          const DataType beta,
          DataType *C
          )
{
    int lda = (TransA == CblasNoTrans)?K:M;
    int ldb = (TransB == CblasNoTrans)?N:K;
    int ldc = N;
    
#ifdef _DOUBLE
    cblas_dgemm(CblasRowMajor,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
#endif
    
#ifdef _FLOAT
    cblas_sgemm(CblasRowMajor,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
#endif
}

void gemv(
          const enum CBLAS_TRANSPOSE Trans,
          const int M,
          const int N,
          const int K,
          const DataType alpha,
          const DataType *A,
          const DataType *x,
          const DataType beta,
          DataType *y
          )
{
    int lda = (Trans == CblasNoTrans)?N:M;
#ifdef _DOUBLE
    cblas_dgemv(CblasRowMajor,Trans,M,N,alpha,A,lda,x,1,beta,y,1);
#endif
    
#ifdef _FLOAT
    cblas_sgemv(CblasRowMajor,Trans,M,N,alpha,A,lda,x,1,beta,y,1);
#endif
}

DataType dot(const int N, const DataType *x, const DataType *y)
{
#ifdef _DOUBLE
    return cblas_ddot(N,x,1,y,1);
#endif
#ifdef _FLOAT
    return cblas_sdot(N,x,1,y,1);
#endif
}

DataType nrm2(const int N, const DataType *x)
{
#ifdef _DOUBLE
    return cblas_dnrm2(N,x,1);
#endif
#ifdef _FLOAT
    return cblas_snrm2(N,x,1);
#endif
}
void copy(const int N, const DataType *x, DataType *y)
{
#ifdef _DOUBLE
    cblas_dcopy(N, x, 1, y, 1);
#endif
#ifdef _FLOAT
    cblas_scopy(N,x,1,y,1);
#endif
}
void swap(const int N, DataType *x, DataType *y)
{
#ifdef _DOUBLE
    cblas_dswap(N, x, 1, y, 1);
#endif
#ifdef _FLOAT
    cblas_sswap(N, x, 1, y, 1);
#endif
}
void scal(const int N, DataType alpha,DataType *x)
{
#ifdef _DOUBLE
    cblas_dscal(N, alpha, x, 1);
#endif
#ifdef _FLOAT
    cblas_sscal(N, alpha, x, 1);
#endif
}
void axpy(const int N, const DataType alpha, const DataType *x, DataType *y)
{
#ifdef _DOUBLE
    cblas_daxpy(N, alpha, x, 1, y, 1);
#endif
#ifdef _FLOAT
    cblas_saxpy(N, alpha, x, 1, y, 1);
#endif
}
void add_n_rows(const int rows, const int N,const DataType *x, DataType *A)
{
    int i = 0;
    for(i = 0;i<rows; i++){
#ifdef _DOUBLE
        daxpy(N, 1.0, x, A);
#endif
#ifdef _FLOAT
        saxpy(N, 1.0, x, A);
#endif
        A += N;
    }
}
void copy_n_rows(const int rows, const int N, const DataType *x, DataType *A)
{
    int i = 0;
    for(i = 0;i < rows; i++){
#ifdef _DOUBLE
        dcopy(N, x, A);
#endif
#ifdef _FLOAT
        scopy(N, x, A);
#endif
        A += N;
    }
}
