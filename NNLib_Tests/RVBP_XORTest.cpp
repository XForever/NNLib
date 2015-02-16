//
//  XORTest.cpp
//  NN_proj
//
//  Created by vsnRain on 03/10/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#include "RVBP_XORTest.h"

void RVBP_XORTest(){
    
    // ===== CREATING NETWORK =====
    Network *net = new Network();
    net->setTransferType(tanhTransferType);
    
    net->addLayer();
    net->layers.back()->addNeurons(10);
    
    net->addLayer();
    net->layers.back()->addNeurons(2);
    
    
    // ===== CREATING TRAINER =====
    RVBPTrainer *t = new RVBPTrainer();
    t->setCycles(100);
    t->setLearningRate(0.5);
    
    // ===== INIT DATA =====
    rvmatrix trainingData= {
        {1, 1},
        {0, 1},
        {1, 0},
        {0, 0},
    };
    
    vector<vector<int>> targetNeurons = {
        {0, 3},
        {1, 2}
    };
    
    t->setTrainingData(trainingData);
    t->setTargetNeurons(targetNeurons);
    
    // ===== TRAINING =====
    t->train(net);
    
    // ===== TESTING =====
    net->setInput(trainingData.at(0));
    net->run();
    net->printOutput();
    
    net->setInput(trainingData.at(1));
    net->run();
    net->printOutput();
    
    net->setInput(trainingData.at(2));
    net->run();
    net->printOutput();
    
    net->setInput(trainingData.at(3));
    net->run();
    net->printOutput();
    
    // ===== CLEANUP =====
    delete t;
    delete net;
    
}


