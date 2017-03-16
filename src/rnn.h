//
//  rnn.h
//  rnn_test
//
//  Created by xuzhuoyi on 23/12/2016.
//  Copyright © 2016 xuzhuoyi. All rights reserved.
//

#ifndef rnn_h
#define rnn_h

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "structs.h"
#include "blas_ext.h"

/*=========net initialization========*/
void create_net(Nets *pNet, char *pConfigFile);
void release_net(Nets *pNet);

/*=======parameter initialization=====*/
void set_value(DataType *data,DataType val,int num);
void set_zero(DataType *data,int num);
void initialize(Nets *pNet);
void random_initialize_weights(Nets *pNet);

/*========unit function in net==========*/
DataType sigmoid(DataType val);
DataType _tanh(DataType val);
DataType ReLU(DataType val);
void activation(DataType *pData,int length,DataType (*non_linear)(DataType));
DataType add(DataType a,DataType b);
DataType sub(DataType a,DataType b);
DataType mul(DataType a,DataType b);
void tanh_batch(DataType *A,DataType *B,int N);

DataType dsigmoid(DataType val);
DataType dtanh(DataType val);
DataType dReLU(DataType val);
void dactivation(DataType *pSrcData,DataType *pDstData,int length, DataType(*derivation)(DataType));

void elementWise(DataType *A,DataType *B,DataType *C,int N,DataType (*op)(DataType,DataType));

void net_out(DataType *h,DataType *y,DataType *W,DataType *b,int nSamples,int nSeqs,int nhiddens,int nYs,DIRECTION_TYPE direction,int isInput);

void lstm_forward(DataType *x, DataType *h,DataType *input,DataType *output,DataType *forget,DataType *cell,DataType *cell_hat,DataType *cell_tanh,DataType *W,DataType *U,DataType *b,int nSamples,int nSeqs,int nFeatures,int nHiddens,DIRECTION_TYPE direction,int isInputLayer);
void forward(DataType *pFeat,Nets *pNet,int nSamples,int nSeqs);

void lstm_backward(DataType *dEdh,DataType *dEdct,DataType *dEdo,DataType *dEdi,DataType *dEdf,DataType *dEdc,DataType *dEdch,DataType *dEdx,DataType *dEdWi,DataType *dEdUi,DataType *dEdWf,DataType *dEdUf,DataType *dEdWo,DataType *dEdUo,DataType *dEdWc,DataType *dEdUc,DataType *dEdbi,DataType *dEdbf,DataType *dEdbo,DataType *dEdbc,DataType *x,DataType *input,DataType *forget,DataType *output,DataType *cell,DataType *cell_hat,DataType *cell_tanh,DataType *h,DataType *Wi,DataType *Ui,DataType *Wf,DataType *Uf,DataType *Wo,DataType *Uo,DataType *Wc,DataType *Uc,int nSeqs,int nSamples,int nHiddens,int nFeatures,DIRECTION_TYPE direction,int isInput);

DataType bp(DataType *pFeat,DataType *pLabel,Nets *pNet, Nets *pNetbp,int nSamples,int nSeqs);
void sgd(Nets *pNet,Nets *pNetbp,DataType lamda,DataType lr,int nSamples);
void check_gradient(DataType *pFeature,DataType *pLabel,Nets *pNet,DataType dw,int nWeightId,int nSamples,int nSeqs);
DataType test(DataType *test);

#endif /* rnn_h */
