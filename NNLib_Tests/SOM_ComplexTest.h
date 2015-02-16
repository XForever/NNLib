//
//  SOM_ComplexTest.h
//  NN_proj
//
//  Created by vsnRain on 15/10/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#ifndef __NN_proj__SOM_ComplexTest__
#define __NN_proj__SOM_ComplexTest__

#include "CVSOMTrainer.h"

void SOM_ComplexTest();

void prepareData(cvmatrix &trainingData);
double magnitude(complex<double> arg);
void dumpResult(Network *net, cvmatrix trainingData);

#endif /* defined(__NN_proj__SOM_ComplexTest__) */
