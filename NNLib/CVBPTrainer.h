//
//  CVBPTrainer.h
//  NN_proj
//
//  Created by vsnRain on 14/10/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#ifndef __NN_proj__CVBPTrainer__
#define __NN_proj__CVBPTrainer__

#include "Trainer.h"

class CVBPTrainer : public Trainer {
    
private:
    int reqNeurons;
    
    cvmatrix trainingData;
    vector<vector<int>> targetNeurons;
    vector<string> targetValues;
    
public:
    void setTrainingData(cvmatrix data);
    void setTargetNeurons(vector<vector<int>> target);
    
    CVBPTrainer();
    
    void train(Network *n);
    
    complex<double> errorDerivative(complex<double> val);
};

#endif /* defined(__NN_proj__CVBPTrainer__) */
