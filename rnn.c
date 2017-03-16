//
//  rnn.c
//  rnn_test
//
//  Created by xuzhuoyi on 23/12/2016.
//  Copyright © 2016 xuzhuoyi. All rights reserved.
//

#include "rnn.h"

//
//一维LSTM输入的维度为向量，nfeatures为输入某个时刻的维度
///////////////initialize the network
//
//
void create_net(Nets *pNet, char *pConfigFile)
{
    FILE* pFile = fopen(pConfigFile,"r");
    if(!pFile){
        printf("Fail to load the file");
    }
    
    int tmp = 0;
    fscanf(pFile,"%d",&tmp);
    switch(tmp){
        case 0:
            pNet->net_type = RNN_SIMPLE;
            break;
        case 1:
            pNet->net_type = RNN_LSTM;
            break;
        case 2:
            pNet->net_type = RNN_GRU;
            break;
        default:
            break;
    }
    
    fscanf(pFile,"%d",&(pNet->nLayers));
    fscanf(pFile,"%d",&(pNet->nSamples)); //batch
    fscanf(pFile,"%d",&(pNet->nMaxSeqs)); //sequence length
    fscanf(pFile,"%d",&(pNet->nFeatures)); //input dimension
    
    pNet->pLayer = (Layer*)malloc(sizeof(Layer)*(pNet->nLayers));
    
    fscanf(pFile,"%d",&tmp);
    if(tmp){
        pNet->direction = UNIDIRECTION;
    }
    else{
        pNet->direction = BIDIRECTION;
    }
    
    int nWeights = 0;
    int nBias = 0;
    
    int i = 0;
    int j = 0;
    int id = 0;
    
    Layer* pLayer = pNet->pLayer;
    
    for(i =0; i<pNet->nLayers;i++){
        fscanf(pFile,"%d",&id);
        fscanf(pFile,"%d",&(pLayer->nNodes));
        fscanf(pFile,"%d",&(pLayer->nPrevs));
        pLayer->pPrev = (int *)malloc(sizeof(int)*pLayer->nPrevs);
        
        for(j=0;j<pLayer->nPrevs;j++){
            fscanf(pFile,"%d",&(pLayer->pPrev[j]));
        }
        
        fscanf(pFile,"%d",&tmp);
        switch(tmp){
            case 0:
                pLayer->nLayerType = RECURRENT;
                break;
            case 1:
                pLayer->nLayerType = FC;
                break;
            default:
                break;
        }
        
        switch(pLayer->nLayerType){
            case RECURRENT:
                switch(pNet->net_type){
                    case RNN_SIMPLE:
                        ;
                        break;
                    case RNN_LSTM:
                        switch(pNet->direction){
                            case UNIDIRECTION:
                                pLayer->nbs = 4*pLayer->nNodes;  //forget gate,input gate, output gate,cell hat
                                pLayer->nhs = pNet->nSamples*pNet->nMaxSeqs*pLayer->nNodes;
                                pLayer->nWs = 4*pLayer->nNodes*pLayer->nNodes; //multiply ht-1, so input and output have the same dimension
                                if(pLayer->pPrev[0]==-1){ //if input layer
                                    pLayer->nUs = 4*pLayer->nNodes*pNet->nFeatures;
                                }
                                else{
                                    pLayer->nUs = 4*pLayer->nNodes*pNet->pLayer[pLayer->pPrev[0]].nNodes;
                                }
                                pLayer->pCell = (DataType *)malloc(sizeof(DataType)*pLayer->nhs); //cell
                                pLayer->pCell_h = (DataType *)malloc(sizeof(DataType)*pLayer->nhs); //hat
                                pLayer->pInput = (DataType *)malloc(sizeof(DataType)*pLayer->nhs);
                                pLayer->pCell_t = (DataType *)malloc(sizeof(DataType)*pLayer->nhs); //Ct
                                pLayer->pForget = (DataType *)malloc(sizeof(DataType)*pLayer->nhs);
                                pLayer->pOutput = (DataType *)malloc(sizeof(DataType)*pLayer->nhs);
                                pLayer->ph = (DataType *)malloc(sizeof(DataType)*pLayer->nhs); //hidden
                                
                                nWeights += pLayer->nWs;
                                nWeights += pLayer->nUs;
                                nBias += pLayer->nbs;
                                break;
                            case BIDIRECTION:
                                ;
                                break;
                            default:
                                break;
                        }
                        break;
                    case RNN_GRU:
                        ;
                        break;
                    default:
                        break;
                }
                break;
            case FC:
                switch (pNet->direction) {
                    case UNIDIRECTION:
                        pLayer->nbs = pLayer->nNodes;
                        pLayer->nWs = 0;
                        pLayer->nhs = pNet->nSamples*pNet->nMaxSeqs*pLayer->nNodes;
                        if(pLayer->pPrev[0] == -1){
                            pLayer->nUs = pNet->nFeatures*pLayer->nNodes;
                        }
                        else{
                            pLayer->nUs = pLayer->nNodes*pLayer[pLayer->pPrev[0]].nNodes;
                        }
                        pLayer->pCell_t = NULL;
                        pLayer->pCell_h = NULL;
                        pLayer->pCell = NULL;
                        pLayer->pOutput = NULL;
                        pLayer->pInput = NULL;
                        pLayer->pForget = NULL;
                        pLayer->ph = (DataType*)malloc(sizeof(DataType)*pLayer->nhs);
                        
                        nWeights += pLayer->nhs;
                        nBias += pLayer->nbs;
                        
                        break;
                    case BIDIRECTION:
                        ;
                        break;
                    default:
                        break;
                }
                break;
            default:
                break;
        }
        pLayer++;
    }
    pNet->nWeights = nWeights;
    pNet->nBias = nBias;
    pNet->pWeight = (DataType *)malloc(sizeof(DataType)*(pNet->nWeights + pNet->nBias));
    pNet->pBias = pNet->pWeight + pNet->nWeights;
    //pNet->pBias = (DataType *)malloc(sizeof(DataType)*pNet->nBias);
    pLayer = pNet->pLayer;
    
    int nWoffset = 0;
    int nBoffset = 0;
    
    for(i = 0;i<pNet->nLayers;i++){
        switch (pLayer->nLayerType) {
            case RECURRENT:
                switch(pNet->net_type){
                    case RNN_SIMPLE:
                        ;
                        break;
                    case RNN_LSTM:
                        pLayer->pW = pNet->pWeight+nWoffset;  //not malloc memory units but point to the pNet->pWeight,share the same memory storage units
                        nWoffset += pLayer->nWs;
                        pLayer->pU = pNet->pWeight+nWoffset;
                        nWoffset += pLayer->nUs;
                        pLayer->pb = pNet->pBias+nBoffset;
                        nBoffset += pLayer->nbs;
                        break;
                    case RNN_GRU:
                        ;
                        break;
                    default:
                        break;
                }
                break;
            case FC:
                pLayer->pU = pNet->pWeight+nWoffset;
                nWoffset += pLayer->nUs;
                pLayer->pb = pNet->pBias + nBoffset;
                nBoffset += pLayer->nbs;
                break;
            default:
                break;
        }
        pLayer++;
    }
    
    fclose(pFile);
}

void release_net(Nets *pNet){
    
    int i =0;
    Layer *player = pNet->pLayer;
    
    for(i=0;i<pNet->nLayers;i++){
        free(player->pPrev);
        player->pPrev = NULL;
        
        switch(player->nLayerType){
            case RECURRENT:
                switch(pNet->net_type){
                    case RNN_SIMPLE:
                        ;
                        break;
                    case RNN_LSTM:
                        free(player->pCell);
                        player->pCell = NULL;
                        free(player->pInput);
                        player->pInput = NULL;
                        free(player->pOutput);
                        player->pOutput = NULL;
                        free(player->pForget);
                        player->pForget = NULL;
                        free(player->ph);
                        player->ph = NULL;
                        free(player->pCell_h);
                        player->pCell_h = NULL;
                        free(player->pCell_t);
                        player->pCell_t = NULL;
                        
                        player->nbs = 0;
                        player->nhs = 0;
                        player->nNodes = 0;
                        player->nUs = 0;
                        player->nWs = 0;
                        player->pb = NULL;
                        player->pU = NULL;
                        player->pW = NULL;
                        break;
                    case RNN_GRU:
                        break;
                    default:
                        break;
                }
            case FC:
                free(player->ph);
                player->ph = NULL;
                
                player->nWs = 0;
                player->nUs = 0;
                player->nbs = 0;
                player->nNodes = 0;
                player->nhs = 0;
                player->pW = NULL;
                player->pU = NULL;
                player->pb = NULL;
                break;
            default:
                break;
        }
        player++;
    }
    free(pNet->pWeight);
    pNet->pWeight = NULL;
    free(pNet->pBias);
    pNet->pBias = NULL;
    
    pNet->nWeights = 0;
    pNet->nBias = 0;
    
    free(pNet->pLayer);
    pNet->pLayer = NULL;
    pNet->nLayers = 0;
    pNet->nSamples = 0;
    pNet->nMaxSeqs = 0;
}
void set_value(DataType *data,DataType val,int num){
    
    int i = 0;
    for(i=0;i<num;i++){
        *data = val;
        data++;
    }
    
}
void set_zero(DataType *data,int num){
    set_value(data,0,num);
}
void initialize(Nets *pNet){
    int i = 0;
    
    Layer *pLayer = pNet->pLayer;
    
    for(i = 0;i<pNet->nLayers;i++){
        switch(pLayer->nLayerType){
            case RECURRENT:
                switch(pNet->net_type){
                    case RNN_SIMPLE:
                        break;
                    case RNN_LSTM:
                        set_zero(pLayer->ph, pLayer->nhs);
                        set_zero(pLayer->pCell,pLayer->nhs);
                        set_zero(pLayer->pCell_h,pLayer->nhs);
                        set_zero(pLayer->pCell_t,pLayer->nhs);
                        set_zero(pLayer->pInput,pLayer->nhs);
                        set_zero(pLayer->pForget,pLayer->nhs);
                        set_zero(pLayer->pOutput,pLayer->nhs);
                        break;
                    case RNN_GRU:
                        break;
                    default:
                        break;
                }
                break;
            case FC:
                set_zero(pLayer->ph,pLayer->nhs);
                break;
            default:
                break;
        }
        pLayer++;
    }
}
//all weights initialize average-norm
void random_initialize_weights(Nets *pNet){
    int i = 0;
    DataType eps = (DataType)1e-2;
    DataType *ptmpw = pNet->pWeight;
    for(i=0;i<pNet->nWeights;i++){
        int num = rand()%1000;  //why is 1000???
        *ptmpw = (DataType)(num)/1000.0*2*eps-eps;
    }
    set_zero(pNet->pBias,pNet->nBias);
}

//
//
//
//////////////auxiliary functions
//
DataType sigmoid(DataType val)
{
    return 1.0/(1.0+exp(-val));
}
DataType _tanh(DataType val)
{
    return (1.0-exp(-2.0*val))/(1.0+exp(-2.0*val));
}
DataType ReLU(DataType val)
{
    return (val>0)?val:0;
}

void activation(DataType *pData,int length,DataType (*non_linear)(DataType))
{
    int i = 0;
    DataType *pTmp = pData;
    for(i = 0; i<length;i++){
        *pData++ = non_linear(*pTmp++);
    }
}

DataType add(DataType a,DataType b)
{
    return a+b;
}
DataType sub(DataType a,DataType b)
{
    return a-b;
}
DataType mul(DataType a,DataType b)
{
    return a*b;
}
void tanh_batch(DataType *A,DataType *B,int N)
{
    int i = 0;
    for(i = 0;i<N;i++){
        *B++ += _tanh(*A++);
    }
}
void elementWise(DataType *A,DataType *B,DataType *C,int N,DataType (*op)(DataType,DataType))
{
    int i = 0;
    for(i = 0;i<N;i++){
        DataType val1 = *A++;  //equal then increment
        DataType val2 = *B++;  //equal then increment
        *C++ += op(val1,val2);
    }
}
void net_out(DataType *h,DataType *y,DataType *W,DataType *b,int nSamples,int nSeqs,int nhiddens,int nYs,DIRECTION_TYPE direction,int isInput)
{
    switch(direction){
        case UNIDIRECTION:
            copy_n_rows(nSeqs*nSamples, nYs, b, y); //add bias
            gemm(CblasNoTrans, CblasTrans, nSeqs*nSamples, nYs, nhiddens, 1.0, h, W, 1.0, y); //y = y+h*W
            activation(y, nSeqs*nSamples, sigmoid);
            break;
        case BIDIRECTION:
            break;
        default:
            break;
    }
}
//
//
///////////former function
//
//
void lstm_forward(DataType *x, DataType *h,DataType *input,DataType *output,DataType *forget,DataType *cell,DataType *cell_hat,DataType *cell_tanh,DataType *W,DataType *U,DataType *b,int nSamples,int nSeqs,int nFeatures,int nHiddens,DIRECTION_TYPE direction,int isInputLayer)
{
    int M = nSeqs*nSamples;  //not include nhiddens,nseqs is the outside dimension
    
    int b_offset = nHiddens;
    int W_offset = nHiddens*nHiddens;
    int U_offset = nHiddens*nFeatures;
//    int h_offset = nHiddens;
//    int x_offset = nSamples*nSeqs*nFeatures;
    
    int N = nSamples*nHiddens;  //not include nSeqs
    
    int i = 0;
    
    DataType *tmpB = b;
    DataType *tmpU = U;
    DataType *tmpW = W;
    DataType *tmpH = h;
    DataType *tmpC = cell;
    
    switch(direction){
        case UNIDIRECTION:
            //****************t = 0
            //plus bias
            copy_n_rows(M,nHiddens,tmpB,input);
            tmpB += b_offset;
            
            copy_n_rows(M,nHiddens,tmpB,forget);
            tmpB += b_offset;
            
            copy_n_rows(M,nHiddens,tmpB,output);
            tmpB += b_offset;
            
            copy_n_rows(M,nHiddens,tmpB,cell_hat);
            
            //plus Ux
            gemm(CblasNoTrans,CblasTrans,M,nHiddens,nFeatures,1.0,x,tmpU,1.0,input); //C = A*B+C input = input+tmpU*nFeatures
            
            tmpU += U_offset;
            gemm(CblasNoTrans,CblasTrans,M,nHiddens,nFeatures,1.0,x,tmpU,1.0,forget); //forget
            
            tmpU += U_offset;
            gemm(CblasNoTrans,CblasTrans,M,nHiddens,nFeatures,1.0,x,tmpU,1.0,output); //output
            
            tmpU += U_offset;
            gemm(CblasNoTrans,CblasTrans,M,nHiddens,nFeatures,1.0,x,tmpU,1.0,cell_hat); //C~
            
            activation(input,N,sigmoid);
            activation(forget,N,sigmoid);
            activation(output,N,sigmoid);
            activation(cell_hat,N,_tanh);
            
            //Ct = ft*Ct-1 + it*Chat-t
            elementWise(input, cell_hat, cell, N, mul);  //where is cellt-1
            
            //tanh(Ct)
            tanh_batch(cell, cell_tanh, N);
            //h = ot*tanh(Ct)
            elementWise(output, cell_tanh, h, N, mul);
            //这里的x是所有的输入的序列长度，nSeqs*nSamples*nFeatures,下面这个移位只是移动了一个timestep的长度！！注意！！！并不是移动了所有的seqs
            h += N;
            input += N;
            forget += N;
            output += N;
            cell_hat += N;
            cell_tanh += N;
            cell += N;
            //***********begin t = 1 to the end
            //
            for(i = 1;i <nSeqs;i++){
                //参数夸时刻共享
                tmpW = W;  //every element in the sequence have the same weights
                gemm(CblasNoTrans, CblasTrans, nSamples, nHiddens, nHiddens, 1.0, tmpH, tmpW, 1.0, input);
                tmpW += W_offset;
                gemm(CblasNoTrans, CblasTrans, nSamples, nHiddens, nHiddens, 1.0, tmpH, tmpW, 1.0, forget);
                tmpW += W_offset;
                gemm(CblasNoTrans, CblasTrans, nSamples, nHiddens, nHiddens, 1.0, tmpH, tmpW, 1.0, output);
                tmpW += W_offset;
                gemm(CblasNoTrans, CblasTrans, nSamples, nHiddens, nHiddens, 1.0, tmpH, tmpW, 1.0, cell_hat);
                
                activation(input, N, sigmoid);
                activation(forget, N, sigmoid);
                activation(output, N, sigmoid);
                activation(cell_hat, N, _tanh);
                
                elementWise(input, cell_hat, cell, N, mul);
                elementWise(forget, tmpC, cell, N, mul);
                tanh_batch(cell, cell_tanh, N);
                elementWise(output, cell_tanh, h, N, mul);
                tmpH += N;
                tmpC += N;
                h += N;
                input += N;
                forget += N;
                output += N;
                cell_hat += N;
                cell_tanh += N;
                cell += N;
            }
            break;
        case BIDIRECTION:
            break;
        default:
            break;
    }
}
void forward(DataType *pFeat,Nets *pNet,int nSamples,int nSeqs)
{
    int i = 0;
    Layer *pLayer = pNet->pLayer;
    for(i = 0;i<pNet->nLayers;i++){
        switch(pLayer->nLayerType){
            case RECURRENT:
                switch(pNet->net_type){
                    case RNN_SIMPLE:
                        break;
                    case RNN_LSTM:
                        if(pLayer->pPrev[0] == -1){
                            lstm_forward(pFeat, pLayer->ph, pLayer->pInput, pLayer->pOutput, pLayer->pForget, pLayer->pCell, pLayer->pCell_h, pLayer->pCell_t, pLayer->pW, pLayer->pU, pLayer->pb,nSamples, nSeqs, pNet->nFeatures, pLayer->nNodes,pNet->direction, 1);
                        }
                        else{
                            lstm_forward(pNet->pLayer[pLayer->pPrev[0]].ph, pLayer->ph, pLayer->pInput, pLayer->pOutput, pLayer->pForget, pLayer->pCell, pLayer->pCell_h, pLayer->pCell_t, pLayer->pW, pLayer->pU, pLayer->pb,nSamples,nSeqs, pNet->pLayer[pLayer->pPrev[0]].nNodes, pLayer->nNodes,pNet->direction, 0);
                        }
                        break;
                    case RNN_GRU:
                        break;
                    default:
                        break;
                }
                break;
            case FC:
                if(pLayer->pPrev[0] == -1){
                    net_out(pFeat, pLayer->ph, pLayer->pU, pLayer->pb, nSamples, nSeqs, pNet->nFeatures ,pLayer->nNodes,pNet->direction,1);
                }
                else{
                    net_out(pNet->pLayer[pLayer->pPrev[0]].ph, pLayer->ph, pLayer->pU, pLayer->pb, nSamples, nSeqs,pNet->pLayer[pLayer->pPrev[0]].nNodes, pLayer->nNodes, pNet->direction, 0);
                }
                break;
            default:
                break;
        }
        pLayer++;
    }
}
DataType dsigmoid(DataType val)
{
    return val*(1.0-val);
}
DataType dtanh(DataType val)
{
    return 1.0 - val*val;
}
DataType dReLU(DataType val)
{
    return (val>0)?1:0;
}
void dactivation(DataType *pSrcData,DataType *pDstData,int length, DataType(*derivation)(DataType))
{
    int i = 0;
    for(i = 0;i<length;i++){
        *pDstData++ *= derivation(*pSrcData++);
    }
}
void lstm_backward(DataType *dEdh,DataType *dEdct,DataType *dEdo,DataType *dEdi,DataType *dEdf,DataType *dEdc,DataType *dEdch,DataType *dEdx,DataType *dEdWi,DataType *dEdUi,DataType *dEdWf,DataType *dEdUf,DataType *dEdWo,DataType *dEdUo,DataType *dEdWc,DataType *dEdUc,DataType *dEdbi,DataType *dEdbf,DataType *dEdbo,DataType *dEdbc,DataType *x,DataType *input,DataType *forget,DataType *output,DataType *cell,DataType *cell_hat,DataType *cell_tanh,DataType *h,DataType *Wi,DataType *Ui,DataType *Wf,DataType *Uf,DataType *Wo,DataType *Uo,DataType *Wc,DataType *Uc,int nSeqs,int nSamples,int nHiddens,int nFeatures,DIRECTION_TYPE direction,int isInput)
{
    int M = nSeqs*nSamples;
    int b_offset = 4*nHiddens;
    int W_offset = 4*nHiddens*nHiddens;
    int U_offset = (isInput?4:8)*nHiddens*nFeatures;
    int h_offset = M*nHiddens;
    int x_offset = M*nFeatures;
    
    int N = nSamples*nHiddens;
    int L = nSamples*nFeatures;
    
    int i = 0;
    
    DataType *tmpC = cell;
    DataType *tmpH = h;
    DataType *tmpDc = dEdc;
    DataType *tmpDh = dEdh;
    
    DataType *tmpCell = cell;
    DataType *tmpCell_tanh = cell_tanh;
    DataType *tmpCell_hat = cell_hat;
    DataType *tmpForget = forget;
    DataType *tmpOutput = output;
    DataType *tmpHH = h;
    DataType *tmpXX = x;
    
    DataType *tmpdEdh = dEdh;
    DataType *tmpdEdo = dEdo;
    DataType *tmpdEdct = dEdct;
    DataType *tmpdEdi = dEdi;
    DataType *tmpdEdch = dEdch;
    DataType *tmpdEdf = dEdf;
    DataType *tmpdEdx = dEdx;
    
    switch(direction){
        case UNIDIRECTION:
            //first point to the end of vector move to last timestep, then go to front
            cell += h_offset -N;
            cell_tanh += h_offset -N;
            cell_hat += h_offset -N;
            forget += h_offset -N;
            output += h_offset -N;
            input += h_offset -N;
            h += h_offset -N;
            x += x_offset -L;
            
            dEdh += h_offset -N;
            dEdo += h_offset -N;
            dEdc += h_offset -N;
            dEdct += h_offset -N;
            dEdi += h_offset -N;
            dEdch += h_offset -N;
            dEdf += h_offset -N;
            
            if(!isInput){
                dEdx += x_offset -L;
            }
            
            tmpDc = dEdc - N; //one timestep before
            tmpC = cell - N;
            tmpH = h - N;
            tmpDh = dEdh - N;
            
            for(i = 0;i<nSeqs;i++){
                elementWise(dEdh, cell_tanh, dEdo, N, mul);  //dEdOt
                elementWise(dEdh, output, dEdct, N, mul);  //dEdCt * Ot
                dactivation(cell_tanh, dEdct, N, dtanh); //dEdCt*Ot  * dtanh(Ct)
                
                axpy(N, 1.0, dEdct, dEdc);  //dEdC = dEdC + dEdCt
                elementWise(dEdc, cell_hat, dEdi, N, mul); //dEdi = dEdC * cell_hat
                elementWise(dEdc, input, dEdch, N, mul);  //dEdChat = input * dEdC
                
                if(i < nSeqs - 1){  //except t = 0
                    elementWise(dEdc, forget, tmpDc, N, mul);  //not understand yet
                    elementWise(dEdc, tmpC, dEdf, N, mul);
                }
                dactivation(input, dEdi, N, dsigmoid);  //dEdi = dsig(it) for the preparation for wi and ui
                if(i < nSeqs - 1){
                    dactivation(forget, dEdf, N, dsigmoid);  //dEdf = dsig(ft)
                }
                dactivation(output, dEdo, N, dsigmoid);  //dEdo = dsig(Ot)
                dactivation(cell_hat, dEdch, N, dtanh); //dEdch(tanh) = dsig(cell_hat)
                
                if(i < nSeqs - 1){
                    gemm(CblasTrans, CblasNoTrans, nHiddens, nHiddens, nSamples, 1.0, dEdi, tmpH, 1.0, dEdWi);  //it = sig(wi*ht-1 + ui*xt + bi)
                    gemm(CblasTrans, CblasNoTrans, nHiddens, nHiddens, nSamples, 1.0, dEdf, tmpH, 1.0, dEdWf);
                    gemm(CblasTrans, CblasNoTrans, nHiddens, nHiddens, nSamples, 1.0, dEdo, tmpH, 1.0, dEdWo);
                    gemm(CblasTrans, CblasNoTrans, nHiddens, nHiddens, nSamples, 1.0, dEdch, tmpH, 1.0, dEdWc);
                }
                if(i < nSeqs - 1){
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nHiddens, nHiddens, 1.0, dEdi, Wi, 1.0, dEdh);
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nHiddens, nHiddens, 1.0, dEdf, Wf, 1.0, dEdh);
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nHiddens, nHiddens, 1.0, dEdo, Wo, 1.0, dEdh);
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nHiddens, nHiddens, 1.0, dEdch, Wc, 1.0, dEdh);
                }
                gemm(CblasTrans, CblasNoTrans, nHiddens, nFeatures, nSamples, 1.0, dEdi, x, 1.0, dEdUi);
                if(i < nSeqs - 1){
                    gemm(CblasTrans, CblasNoTrans, nHiddens, nFeatures, nSamples, 1.0, dEdf, x, 1.0, dEdUf);
                }
                gemm(CblasTrans, CblasNoTrans, nHiddens, nFeatures, nSamples, 1.0, dEdo, x, 1.0, dEdUo);
                gemm(CblasTrans, CblasNoTrans, nHiddens, nFeatures, nSamples, 1.0, dEdch, x, 1.0, dEdUc);
                
                if(!isInput){ //dEdx is cancha theta
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nFeatures, nHiddens, 1.0, dEdi, Ui, 1.0,dEdx);
                    if(i < nSeqs - 1){
                        gemm(CblasNoTrans, CblasNoTrans, nSamples, nFeatures, nHiddens, 1.0, dEdf, Uf, 1.0, dEdx);
                    }
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nFeatures, nHiddens, 1.0, dEdo, Uo, 1.0, dEdx);
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nFeatures, nHiddens, 1.0, dEdch, Uc, 1.0, dEdx);
                }
                add_n_rows(nSamples, nHiddens, dEdi, dEdbi);
                if(i < nSeqs - 1){
                    add_n_rows(nSamples, nHiddens, dEdf, dEdbf);
                }
                add_n_rows(nSamples, nHiddens, dEdo, dEdbo);
                add_n_rows(nSamples, nHiddens, dEdch, dEdbc);
                
                cell_tanh -= N;
                cell_hat -= N;
                cell -= N;
                input -= N;
                forget -= N;
                input -= N;
                output -= N;
                h -= N;
                x -= L;  //if is input layer?
                
                dEdh -= N;
                dEdo -= N;
                dEdf -= N;
                dEdc -= N;
                dEdct -= N;
                dEdch -= N;
                dEdi -= N;
                if(!isInput){
                    dEdx -= L;
                }
                if(i < nSeqs - 1){ //not t=0
                    tmpDc -= N;
                    tmpC -= N;
                    tmpH -= N;
                    tmpDh -= N;
                }
            }
            break;
        case BIDIRECTION:
            break;
        default:
            break;
    }
}
DataType delta_loss(DataType *dEdy,DataType *dEdW,DataType *dEdb,DataType *dEdh,DataType *y,DataType *label,DataType *h,DataType *W,int nSeqs,int nSamples,int nHiddens,int nYs,DIRECTION_TYPE direction,int isInput)
{
    int offset1 = nSamples*nYs;  //nys output
    //int offset = nSamples*nHiddens;
    DataType *tmpH = h;
    DataType *tmpDy = dEdy;
    DataType cost = 0;
    DataType val = 0;
    
    switch(direction){
        case UNIDIRECTION:
            copy(nSeqs*offset1, y, dEdy); //copy y to dEdy, y is output
            axpy(nSeqs*offset1,-1.0, label, dEdy);  //dEdy = y -label
            val = nrm2(nSeqs*offset1, dEdy); //normalize
            cost += val*val/2.0;  //(y-label)^2 * 0.5
            dactivation(y, dEdy, nSeqs*offset1, dsigmoid); //dEdy
            gemm(CblasTrans, CblasNoTrans, nYs, nHiddens, nSeqs*nSamples, 1.0, tmpDy, tmpH, 1.0, dEdW); //dEdW += Dy * ht
            add_n_rows(nSeqs*nSamples, nYs, tmpDy, dEdb);  //dEdb += dEdy
            if(!isInput){
                gemm(CblasNoTrans, CblasNoTrans, nSeqs*nSamples, nHiddens, nYs, 1.0, dEdy, W, 1.0, dEdh);
            }
            break;
        case BIDIRECTION:
            break;
        default:
            break;
    }
    return cost;
}
DataType bp(DataType *pFeat,DataType *pLabel,Nets *pNet, Nets *pNetbp,int nSamples,int nSeqs)
{
    initialize(pNet);
    initialize(pNetbp);
    
    DataType cost = 0;
    int i =0;
    
    forward(pFeat, pNet, nSamples, nSeqs);
    
    Layer *pLayer = pNet->pLayer+pNet->nLayers-1;
    Layer *pLayerBP = pNetbp->pLayer+pNetbp->nLayers-1;
    
    for(i = 0;i<pNet->nLayers;i++){
        switch(pLayer->nLayerType){
            case RECURRENT:
                switch(pNet->net_type){
                    case RNN_SIMPLE:
                        break;
                    case RNN_LSTM:
                        if(pLayer->pPrev[0] == -1){
                            lstm_backward(pLayerBP->ph, pLayerBP->pCell_t, pLayerBP->pOutput, pLayerBP->pInput, pLayerBP->pForget, pLayerBP->pCell, pLayerBP->pCell_h, NULL, pLayerBP->pW, pLayerBP->pU, pLayerBP->pW+pLayerBP->nNodes*pLayerBP->nNodes, pLayerBP->pU+pNet->nFeatures*pLayerBP->nNodes, pLayerBP->pW+2*pLayerBP->nNodes*pLayerBP->nNodes, pLayerBP->pU+2*pLayerBP->nNodes*pLayerBP->nNodes, pLayerBP->pW+3*pLayerBP->nNodes*pLayerBP->nNodes, pLayerBP->pU+3*pNet->nFeatures*pLayerBP->nNodes, pLayerBP->pb, pLayerBP->pb+pLayerBP->nNodes, pLayerBP->pb+2*pLayerBP->nNodes, pLayerBP->pb+3*pLayerBP->nNodes, pFeat, pLayer->pInput, pLayer->pForget, pLayer->pOutput, pLayer->pCell, pLayer->pCell_h, pLayer->pCell_t, pLayer->ph, pLayer->pW, pLayer->pU, pLayer->pW+pLayer->nNodes*pLayer->nNodes, pLayer->pU+pNet->nFeatures*pLayer->nNodes, pLayer->pW+2*pLayer->nNodes*pLayer->nNodes, pLayer->pU+2*pNet->nFeatures*pLayer->nNodes, pLayer->pW+3*pLayer->nNodes*pLayer->nNodes, pLayer->pU+pNet->nFeatures*pLayer->nNodes, nSeqs, nSamples, pLayer->nNodes, pNet->nFeatures, pNet->direction, 1);
                        }
                        else{
                            lstm_backward(pLayerBP->ph, pLayerBP->pCell_t, pLayerBP->pOutput, pLayerBP->pInput, pLayerBP->pForget, pLayerBP->pCell, pLayerBP->pCell_h,pNetbp->pLayer[pLayerBP->pPrev[0]].ph, pLayerBP->pW, pLayerBP->pU, pLayerBP->pW+pLayerBP->nNodes*pLayerBP->nNodes, pLayerBP->pU+pNetbp->pLayer[pLayerBP->pPrev[0]].nNodes*pLayerBP->nNodes, pLayerBP->pW+2*pLayerBP->nNodes*pLayerBP->nNodes, pLayerBP->pU+2*pNetbp->pLayer[pLayerBP->pPrev[0]].nNodes*pLayerBP->nNodes, pLayerBP->pW+3*pLayerBP->nNodes*pLayerBP->nNodes, pLayerBP->pU+3*pNetbp->pLayer[pLayerBP->pPrev[0]].nNodes, pLayerBP->pb, pLayerBP->pb+pLayerBP->nNodes, pLayerBP->pb+2*pLayerBP->nNodes, pLayerBP->pb+3*pLayerBP->nNodes, pNet->pLayer[pLayer->pPrev[0]].ph, pLayer->pInput, pLayer->pForget, pLayer->pOutput, pLayer->pCell, pLayer->pCell_h, pLayer->pCell_t, pLayer->ph, pLayer->pW, pLayer->pU, pLayer->pW+pLayer->nNodes*pLayer->nNodes, pLayer->pU+pNet->pLayer[pLayer->pPrev[0]].nNodes*pLayer->nNodes, pLayer->pW+2*pLayer->nNodes*pLayer->nNodes, pLayer->pU+2*pNet->pLayer[pLayer->pPrev[0]].nNodes*pLayer->nNodes, pLayer->pW+3*pLayer->nNodes*pLayer->nNodes, pLayer->pU+3*pNet->pLayer[pLayer->pPrev[0]].nNodes*pLayer->nNodes, nSeqs, nSamples, pLayerBP->nNodes, pNetbp->pLayer[pLayerBP->pPrev[0]].nNodes, pNet->direction, 0);
                        }
                        break;
                    case RNN_GRU:
                        break;
                    default:
                        break;
                }
                break;
            case FC:
                if(pLayer->pPrev[0] == -1){
                    cost+= delta_loss(pLayerBP->ph, pLayerBP->pU, pLayerBP->pb, NULL,pLayer->ph, pLabel, pFeat, pLayer->pU, nSeqs, nSamples, pNet->nFeatures, pLayer->nNodes, pNet->direction, 1);
                }
                else{
                    cost += delta_loss(pLayerBP->ph, pLayerBP->pU, pLayerBP->pb, pNetbp->pLayer[pLayerBP->pPrev[0]].ph, pLayer->ph, pLabel, pNet->pLayer[pLayer->pPrev[0]].ph, pLayer->pU, nSeqs, nSamples,pNet->pLayer[pLayer->pPrev[0]].nNodes, pLayer->nNodes, pNetbp->direction, 0);
                }
                break;
            default:
                break;
        }
        pLayer--;
        pLayerBP--;
    }
    return cost;
}
void sgd(Nets *pNet,Nets *pNetbp,DataType lamda,DataType lr,int nSamples)
{
    //pweight pbias is after memory next
    scal(pNetbp->nWeights+pNetbp->nBias, 1.0/(DataType)nSamples, pNetbp->pWeight); //w-grad = 1/batch * w-grad
    axpy(pNetbp->nWeights, lamda, pNet->pWeight, pNetbp->pWeight); //bp-weight = weight *lamda + bp-weight
    axpy(pNetbp->nWeights+pNetbp->nBias, -lr, pNetbp->pWeight, pNet->pWeight); //weight = weight - bpweight*lr
}
void check_gradient(DataType *pFeature,DataType *pLabel,Nets *pNet,DataType dw,int nWeightId,int nSamples,int nSeqs)
{
    DataType eps = 1e-4;
    DataType *py = (DataType *)malloc(pNet->pLayer[pNet->nLayers-1].nhs*sizeof(DataType)); //last layer nhs = nseq*nsamples*nodes
    
    DataType W = *(pNet->pWeight+nWeightId);
    
    DataType cost1 = 0;
    DataType cost2 = 0;
    DataType val = 0;
    
    pNet->pWeight[nWeightId] = W-eps;
    initialize(pNet);  //initialize the gate value not weight
    forward(pFeature, pNet, nSamples, nSeqs);
    
    set_zero(py, pNet->pLayer[pNet->nLayers-1].nhs);
    copy(pNet->pLayer[pNet->nLayers-1].nhs,pNet->pLayer[pNet->nLayers-1].ph,py);
    axpy(pNet->pLayer[pNet->nLayers-1].nhs,-1.0,pLabel,py); //py = py - label
    
    val = nrm2(pNet->pLayer[pNet->nLayers-1].nhs,py);
    cost2 += val*val/2.0;
    
    pNet->pWeight[nWeightId] = W+eps;
    initialize(pNet);
    forward(pFeature,pNet,nSamples,nSeqs);
    set_zero(py, pNet->pLayer[pNet->nLayers-1].nhs);
    copy(pNet->pLayer[pNet->nLayers-1].nhs, pNet->pLayer[pNet->nLayers-1].ph, py);
    axpy(pNet->pLayer[pNet->nLayers-1].nhs,-1.0,pLabel,py); //py = py - label
    
    val = nrm2(pNet->pLayer[pNet->nLayers-1].nhs, py);
    cost1 += val*val/2.0;
    
    printf("cost1 = %f\t,cost2 = %f\n",cost1,cost2);
    DataType dw_true = (cost1 - cost2)/(2*eps);
    
    if(fabs(dw_true-dw) < 1e-3){
        printf("success\n");
    }
    else{
        printf("failed");
    }
    free(py);
    py = NULL;
}
DataType test(DataType *test){
    return *test;
}
