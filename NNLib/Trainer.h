//
//Trainer.h
//  NN_proj
//
//  Created by vsnRain on 25/08/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#ifndef __NN_proj__Trainer__
#define __NN_proj__Trainer__

#include "Network.h"

class Network;

class Trainer {

protected:
    Network *network;
    
    int cycles;
    double learningRate;

public:
    Trainer();
    
    void setCycles(int c);
    void setLearningRate(double r);
    
    //virtual void setTrainingData();
    //virtual void setTargetNeurons();
    
    virtual void train(Network *n) = 0;
};

#endif
