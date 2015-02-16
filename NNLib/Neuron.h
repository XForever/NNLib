//
//  Neuron.h
//  NN_proj
//
//  Created by vsnRain on 22/08/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#ifndef __NN_proj__Neuron__
#define __NN_proj__Neuron__

#include <fstream>
#include <math.h>

#include "Layer.h"
#include "Trainer.h"

class Layer;

class Neuron {
private:
    complex<double> output;
    
    int inputSize;
    
    static complex<double> sigmoidFunction(complex<double> val);
    static complex<double> bipolarSigmoidFunction(complex<double> val);
    static complex<double> tanhFunction(complex<double> val);
    static double magnitude(complex<double> comp);
    
public:
    
    Layer *layer;
    int tag;
    complex<double> error;
    
    vector<complex<double>> inputLayer;
    vector<complex<double>> weights;
    complex<double> bias_weight;
    
    Neuron();
    ~Neuron();
    void initWeights();
    
    void setLayer(Layer *l);
    void setInputLayer(vector<complex<double>> inp);
    void setTag(int t);
    
    void check();
    complex<double> count();
    complex<double> activate();
    complex<double> getOutput();
    
    void printWeights();
    void dumpWeightsArray();
    void dumpWeightsMatrix();
    
    void description();
};

#endif