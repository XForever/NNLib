//
//  ImageTest.h
//  NN_proj
//
//  Created by vsnRain on 03/10/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#ifndef __NN_proj__ImageTest__
#define __NN_proj__ImageTest__

#include "RVBPTrainer.h"
#include <set>

void Perceptron_ImageTest();
void prepareData(rvmatrix &trainingData, vector<vector<int>> &targetNeurons);
vector<double> extractData(string file);
void dumpWeights(Network *net);

#endif /* defined(__NN_proj__ImageTest__) */
