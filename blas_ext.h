//
//  blas_ext.h
//  rnn_test
//
//  Created by xuzhuoyi on 2016/12/30.
//  Copyright © 2016年 xuzhuoyi. All rights reserved.
//

#ifndef blas_ext_h
#define blas_ext_h

#include <stdio.h>
#include "blas.h"
#include "structs.h"


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
          );

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
);

DataType dot(const int N, const DataType *x, const DataType *y);
DataType nrm2(const int N, const DataType *x);

void swap(const int N, DataType *x,DataType *y);
void copy(const int N, const DataType *x, DataType *y);
void axpy(const int N, const DataType alpha, const DataType *x, DataType *y);
void scal(const int N, const DataType alpha, DataType *x);
void copy_n_rows(const int rows, const int N, const DataType *x, DataType *A);
void add_n_rows(const int rows, const int N, const DataType *x, DataType *A);

#endif /* blas_ext_h */
