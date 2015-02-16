//
//  RVSOMTrainer.h
//  NN_proj
//
//  Created by vsnRain on 06/10/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#ifndef __NN_proj__RVSOMTrainer__
#define __NN_proj__RVSOMTrainer__

#include "Trainer.h"
#include <pthread.h>

class RVSOMTrainer : public Trainer {
    
private:
    int reqNeurons;
    
    rvmatrix trainingData;
public:
    void setTrainingData(rvmatrix data);
    //void setRadius(int r);
    
    RVSOMTrainer();
    
    void train(Network *n);
    
    void updateSingleWeight(int nodeNumber, int signalNumber, int iter);
    void updateWeights(int nodeNumber, int signalNumber, int iter);
    
    double learningRate(int iter);
    int radius(int iter);
};

#endif /* defined(__NN_proj__RVSOMTrainer__) */
