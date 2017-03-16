//
//  structs.h
//  rnn_test
//
//  Created by xuzhuoyi on 23/12/2016.
//  Copyright © 2016 xuzhuoyi. All rights reserved.
//

#ifndef structs_h
#define structs_h

#include<stdio.h>

#define _DOUBLE  //define as double,change here to chage all the datatype and functions

#ifdef _DOUBLE
typedef double DataType;
#endif

#ifdef _FLOAT
typedef float DataType;
#endif

typedef enum RNN_TYPE{
    RNN_SIMPLE,
    RNN_LSTM,
    RNN_GRU
}RNN_TYPE;

typedef enum LAYER_TYPE{
    RECURRENT,
    FC
}LAYER_TYPE;

typedef enum ACTIVATION_TYPE{
    SIGMOID,
    TANH
}ACTIVATION_TYPE;

typedef enum DIRECTION_TYPE{
    UNIDIRECTION,
    BIDIRECTION
}DIRECTION_TYPE;

typedef enum DATA_FORMAT{
    NTF,
    TNF
}DATA_FORMAT;

typedef struct Layer{
    LAYER_TYPE nLayerType;
    DataType *pW;
    DataType *pU;
    DataType *pb;
    DataType *ph;
    DataType *pInput;
    DataType *pForget;
    DataType *pCell_h;  //cell hat
    DataType *pOutput;
    DataType *pCell;
    DataType *pCell_t; //cell t-1
    int nWs;
    int nUs;
    int nbs;
    int nhs;
    int nNodes;
    int nPrevs;
    int *pPrev;
}Layer;

typedef struct Nets{
    RNN_TYPE net_type;
    DIRECTION_TYPE direction;
    DataType *pWeight;
    DataType *pBias;
    Layer *pLayer;
    int nLayers;
    int nSamples;
    int nMaxSeqs;
    int nFeatures;  //input dimension??
    int nWeights;
    int nBias;
}Nets;

#endif /* structs_h */
