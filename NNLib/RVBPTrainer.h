//
//  RVBPTrainer.h
//  NN_proj
//
//  Created by vsnRain on 09/09/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#ifndef __NN_proj__RVBPTrainer__
#define __NN_proj__RVBPTrainer__

#include "Trainer.h"

class RVBPTrainer : public Trainer {
    
private:
    int reqNeurons;
    
    rvmatrix trainingData;
    vector<vector<int>> targetNeurons;
    vector<string> targetValues;
    
public:
    void setTrainingData(rvmatrix data);
    void setTargetNeurons(vector<vector<int>> target);
    
    RVBPTrainer();
    
    void train(Network *n);
    
    double errorDerivative(double val);
};

#endif
