//
//  CVSOMTrainer.h
//  NN_proj
//
//  Created by vsnRain on 15/10/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#ifndef __NN_proj__CVSOMTrainer__
#define __NN_proj__CVSOMTrainer__

#include "Trainer.h"
#include <pthread.h>

class CVSOMTrainer : public Trainer {
    
private:
    int reqNeurons;
    
    cvmatrix trainingData;
public:
    void setTrainingData(cvmatrix data);
    
    CVSOMTrainer();
    
    void train(Network *n);
    
    void updateSingleWeight(int nodeNumber, int iter, bool weak);
    void updateWeights(int nodeNumber, int iter);
    
    double learningRate(int iter);
    int radius(int iter);
    
    double magnitude(complex<double> arg);
};

#endif /* defined(__NN_proj__CVSOMTrainer__) */
