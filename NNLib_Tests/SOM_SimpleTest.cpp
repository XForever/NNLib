//
//  SOM_SimpleTest.cpp
//  NN_proj
//
//  Created by vsnRain on 06/10/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#include "SOM_SimpleTest.h"

void SOM_SimpleTest(){
    
    // ===== CREATING NETWORK =====
    Network *net = new Network();
    net->setTransferType(stepTransferType);
    
    net->addLayer();
    net->layers.back()->addNeurons(3);
    
    
    // ===== CREATING TRAINER =====
    RVSOMTrainer *t = new RVSOMTrainer();
    t->setCycles(100);
    t->setLearningRate(0.1);
    
    // ===== INIT DATA =====
    rvmatrix trainingData = {
        {1, 1, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 1, 1},
        {0, 0, 1, 0, 0},
    };
    
    t->setTrainingData(trainingData);
    
    // ===== TRAINING =====
    t->train(net);
    
    //vector<double> test = {1, 0, 0, 0};
    //net->setInput(test);
    //net->run();
    
    net->description();
}