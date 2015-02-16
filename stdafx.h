//
//  stdafx.h
//  NN_proj
//
//  Created by vsnRain on 09/09/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#ifndef NN_proj_stdafx_h
#define NN_proj_stdafx_h

#include <iostream>
#include <vector>
#include <complex>
#include <math.h>

#define rvmatrix vector< vector< double > >
#define cvmatrix vector< vector< complex < double > > >

#define sqr(x)              ((x)*(x))
#define MAX(a,b)            ((a) > (b) ? (a) : (b))
#define MIN(a,b)            ((a) < (b) ? (a) : (b))

#define ARC4RANDOM_MAX                  0x100000000

#define threadnum 8

#define DEBUG_TIME                      0
#define DEBUG_NEURON                    0
#define DEBUG_THREADING                 0

#define DEBUG_TRAINER_TIME              1
#define DEBUG_TRAINER_CYCLES            1
#define DEBUG_TRAINER_INPUT             0
#define DEBUG_TRAINER_LAYERS            0
#define DEBUG_TRAINER_NEURON            0
#define DEBUG_TRAINER_THREADING         0
#define DEBUG_TRAINER_DUMP              1

#define DEBUG_TRAINER_SOM_UPDATE        0
#define DEBUG_TRAINER_SOM_UPDATE_DETAIL 0

using namespace std;

typedef enum {
    stepTransferType = 0,
    sigmoidTransferType = 1,
    bipolarSigmoidTransferType = 2,
    tanhTransferType = 3,
    complexType = 4
} TransferFunctionType;

#endif
