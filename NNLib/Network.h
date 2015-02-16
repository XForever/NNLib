//
//  Network.h
//  NN_proj
//
//  Created by vsnRain on 22/08/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#ifndef __NN_proj__Network__
#define __NN_proj__Network__

#include "Layer.h"

class Layer;

class Network {
private:
    vector<complex<double>> inputLayer;
    vector<complex<double>> outputLayer;
    
    bool inited;
    
public:
    
    TransferFunctionType transferType;
    
    vector<Layer *> layers;

    Network();
    ~Network();
    
    void addLayer();
    void setInput(vector<complex<double>> inp);
    void setInput(vector<double> inp2);
    
    void setInputFile(string filename);
    void setTransferType(TransferFunctionType t);
    
    void printOutput();
    vector<complex<double>> getOutput();
    
    void init();
    
    void run();
    
    void description();
};

#endif