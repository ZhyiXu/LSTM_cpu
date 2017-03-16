//
//  blas.h
//  rnn_test
//
//  Created by xuzhuoyi on 29/12/2016.
//  Copyright © 2016 xuzhuoyi. All rights reserved.
//

#ifndef blas_h
#define blas_h

#include <cblas.h>

void dgemm(
           const enum CBLAS_TRANSPOSE TransA,
           const enum CBLAS_TRANSPOSE TransB,
           const int M,
           const int N,
           const int K,
           const double alpha,
           const double *A,
           const double *B,
           const double beta,
           double *C
           
); //matrix*matrix

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

);

void dgemv(
           const enum CBLAS_TRANSPOSE Trans,
           const int M,
           const int N,
           const int K,
           const double alpha,
           const double *A,
           const double *x,
           const double beta,
           double *y
); //matrix*vector

void sgemv(
           const enum CBLAS_TRANSPOSE Trans,
           const int M,
           const int N,
           const int K,
           const float alpha,
           const float *A,
           const float *x,
           const float beta,
           float *y
);

double ddot(const int N, const double *x,const double *y); //dot product return result
double dnrms(const int N, const double *x); //euclidean norm return result

float sdot(const int N,const float *x, const float *y);
float snrms(const int N, const float *x);

void dcopy(const int N, const double *x, double *y); //copy x to y
void scopy(const int N, const float *x,float *y);

void dscal(const int N, const double alpha, double *x); //x = alpha*x
void sscal(const int N, const float alpha,float *x);

void daxpy(const int N, const double alpha, const double *x,double *y); //y = a*x + y
void saxpy(const int N, const float alpha, const float *x, float *y);

void dcopy_n_rows(const int rows,const int N,const double *x, double *A);
void scopy_n_rows(const int rows,const int N,const float *x, float *A);

void dadd_n_rows(const int rows,const int N, const double *x, double *A);
void sadd_n_rows(const int rows, const int N, const float *x, float *A);

void dswap(const int N, double *x, double *y); //swap x and y
void sswap(const int N, float *x, float *y);


#endif /* blas_h */
