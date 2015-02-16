//
//  Layer.h
//  NN_proj
//
//  Created by vsnRain on 22/08/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#ifndef __NN_proj__Layer__
#define __NN_proj__Layer__

#include <thread>
#include <unistd.h>

#include "Neuron.h"
#include "Network.h"

class Network;
class Neuron;

class Layer {
private:
    
public:
    Network *network;
    vector<Neuron *> neurons;
    vector<complex<double>> outputLayer;
    
    complex<double> bias;
    int tag;
    
    void addNeuron();
    void addNeurons(int num);
    void deleteNeuron();
    
    Layer();
    ~Layer();
    void initNeuronsWeight();
    
    void setNetwork(Network *ntw);
    void setTag(int t);
    void setBias(complex<double> b);
    void setInputLayer(vector<complex<double>> inp);
    
    vector<complex<double>> getOutputLayer();
    
    void process();
    
    void description();
};

#endif