//
//  blas.c
//  rnn_test
//
//  Created by xuzhuoyi on 2016/12/30.
//  Copyright © 2016年 xuzhuoyi. All rights reserved.
//

#include "blas.h"

void dgemm(
           const enum CBLAS_TRANSPOSE TransA,
           const enum CBLAS_TRANSPOSE TransB,
           const int M,  //rows of A and C
           const int N,  //colomns of B and C
           const int K,  //colomns of A rows of B
           const double alpha,
           const double *A,  //M*K
           const double *B,  //K*N
           const double beta,
           double *C  //M*N
)
{
    int lda = (TransA == CblasNoTrans)?K:M;  //coloms of A
    int ldb = (TransB == CblasNoTrans)?N:K;
    int ldc = N;
    
    cblas_dgemm(CblasRowMajor,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
}//C := alpha*op( A )*op( B ) + beta*C,

void sgemm(
           const enum CBLAS_TRANSPOSE TransA,
           const enum CBLAS_TRANSPOSE TransB,
           const int M,
           const int N,
           const int K,
           const float alpha,
           const float *A,
           const float *B,
           const float beta,
           float *C
)
{
    int lda = (TransA == CblasNoTrans)?K:M;
    int ldb = (TransB == CblasNoTrans)?N:K;
    int ldc = N;
    cblas_sgemm(CblasRowMajor,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
}

void dgemv(
           const enum CBLAS_TRANSPOSE Trans,
           const int M, //rows of A
           const int N, //colomns of A
           const int K, //
           const double alpha,
           const double *A,
           const double *x,
           const double beta,
           double *y
)
{
    int lda = (Trans == CblasNoTrans)?N:M;
    cblas_dgemv(CblasRowMajor,Trans,M,N,alpha,A,lda,x,1,beta,y,1);
} //y := alpha*A*x + beta*y

void sgemv(
           const enum CBLAS_TRANSPOSE Trans,
           const int M, //rows of A
           const int N, //colomns of A
           const int K, //
           const float alpha,
           const float *A,
           const float *x,
           const float beta,
           float *y
           )
{
    int lda = (Trans == CblasNoTrans)?N:M;
    cblas_sgemv(CblasRowMajor,Trans,M,N,alpha,A,lda,x,1,beta,y,1);
} //y := alpha*A*x + beta*y

//vector manupolation
double ddot(const int N, const double *x, const double *y)
{
    return cblas_ddot(N,x,1,y,1);
}
float sdot(const int N, const float *x, const float *y)
{
    return cblas_sdot(N,x,1,y,1);
}

double dnrm2(const int N, const double *x)
{
    return cblas_dnrm2(N,x,1);
}
float snrm2(const int N, const float *x)
{
    return cblas_snrm2(N,x,1);
}

void dcopy(const int N, const double *x, double *y)
{
    cblas_dcopy(N,x,1,y,1);
}
void scopy(const int N, const float *x, float *y)
{
    cblas_scopy(N,x,1,y,1);
}

void dscal(const int N, const double alpha, double *x)
{
    cblas_dscal(N,alpha,x,1);
}
void sscal(const int N, const float alpha, float *x)
{
    cblas_sscal(N,alpha,x,1);
}

void daxpy(const int N, const double alpha, const double *x, double *y)
{
    cblas_daxpy(N,alpha,x,1,y,1);
}
void saxpy(const int N, const float alpha,const float *x, float *y)
{
    cblas_saxpy(N,alpha,x,1,y,1);
}

void dcopy_n_rows(const int rows, const int N, const double *x, double *A) //copy the same rows
{
    int i = 0;
    for(i = 0; i<rows;i++){
        dcopy(N,x,A);
        A += N;
    }
}
void scopy_n_rows(const int rows, const int N, const float *x, float *A) //copy the same rows
{
    int i = 0;
    for(i = 0; i<rows;i++){
        scopy(N,x,A);
        A += N;
    }
}

void dadd_n_rows(const int rows, const int N, const double *x, double *A)
{
    int i = 0;
    for(i = 0; i< rows;i++){
        daxpy(N,1.0,x,A);
        x += N;
    }
}
void sadd_n_rows(const int rows, const int N, const float *x, float *A)
{
    int i = 0;
    for(i = 0; i< rows;i++){
        saxpy(N,1.0,x,A);
        x += N;
    }
}

void dswap(const int N, double *x,double *y)
{
    cblas_dswap(N,x,1,y,1);
}
void sswap(const int N, float *x,float *y)
{
    cblas_sswap(N,x,1,y,1);
}
