//
//  main.c
//  rnn_test
//
//  Created by xuzhuoyi on 23/12/2016.
//  Copyright © 2016 xuzhuoyi. All rights reserved.
//

//#include <stdio.h>
//#include <cblas.h>
#include "rnn.h"

int main(int argc, const char * argv[]) {
    // insert code here...
    printf("Hello, World!\n");
    srand(1);
    
    DataType pFeature[6] = {1,0,0,1,0,0};
    
    DataType t;
    t = test(pFeature);
    Nets lstm;
    Nets lstmbp;
    
    char pConfigFile[100] = "config.txt";
    create_net(&lstm, pConfigFile);
    create_net(&lstmbp, pConfigFile);
    
    random_initialize_weights(&lstm);
    set_zero(lstmbp.pWeight, lstmbp.nWeights+lstmbp.nBias);
    
    int nWeightId = 168;
    
    DataType cost = bp(pFeature, pFeature+lstm.nSamples*lstm.nFeatures, &lstm, &lstmbp, lstm.nSamples, lstm.nMaxSeqs);
    printf("cost:%f\n",cost);
    
    check_gradient(pFeature, pFeature+lstm.nSamples*lstm.nFeatures, &lstm, lstmbp.pWeight[nWeightId], nWeightId, lstm.nSamples, lstm.nMaxSeqs);
    
    release_net(&lstm);
    release_net(&lstmbp);
    return 0;
}
