//
//  Layer.cpp
//  NN_proj
//
//  Created by vsnRain on 22/08/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#include "Layer.h"

Layer::Layer(){
    bias = 1;
}

Layer::~Layer(){
    for (int i = 0; i<neurons.size(); i++) {
        delete neurons.at(i);
    }
    
}

void Layer::addNeuron(){
    Neuron *n1 = new Neuron();
    n1->setLayer(this);
    neurons.push_back(n1);
    outputLayer.push_back(0);
}

void Layer::addNeurons(int num){
    for (int i=0; i<num; i++) {
        addNeuron();
    }
}

void Layer::deleteNeuron(){
    if (!neurons.empty()){
     neurons.pop_back();
     outputLayer.pop_back();
    }
}

void Layer::setInputLayer(vector<complex<double>> inp){
    for (int i = 0; i<neurons.size(); i++){
        neurons.at(i)->setInputLayer(inp);
    }
}

vector<complex<double>> Layer::getOutputLayer(){
   return outputLayer;
}

void Layer::setNetwork(Network *ntw){
    network = ntw;
}

void Layer::setTag(int t){
    tag = t;
}

void Layer::setBias(complex<double> b){
    bias = b;
}

void Layer::initNeuronsWeight(){
    for (int i = 0; i<neurons.size(); i++){
        neurons.at(i)->initWeights();
    }
}

void Layer::process(){
    //=============== TIME MARK ===============
    std::chrono::time_point<std::chrono::system_clock> start, end;
    if (DEBUG_TIME){
        start = std::chrono::system_clock::now();
        
        printf("\n#############################\n");
        printf("# LAYER  -  TIME MARK DONE  #\n");
        printf("#############################\n\n");
    }
    //=============== INIT OUTPUT ===============
    outputLayer.clear();
    outputLayer = vector<complex<double>>(neurons.size());
    
    //=============== INIT THREADING ===============
    vector<thread> threads;
    int thr_ind = 0;
    
    int neurons_in_layer = (int)neurons.size();
    
    //=============== ITERATE NEURONS BEGIN ===============
    for (int neu_ind = 0; neu_ind<neurons_in_layer; neu_ind++) {
        neurons.at(neu_ind)->setTag(neu_ind);
            
        //=============== THREAD BEGIN ===============
        threads.push_back(thread([=]() -> void {
            
            if (DEBUG_THREADING){
                printf("==THREAD #%d STARTED FOR NEURON #%d\n", thr_ind, neu_ind);
            }
            
            complex<double> nres = neurons.at(neu_ind)->activate();
            
            outputLayer.at(neu_ind) = nres;
            
            if (DEBUG_THREADING){
                //usleep(10000);
                printf("==THREAD %d DONE\n", thr_ind);
            }
        }));
        //=============== THREAD END ===============
        
        if ((thr_ind == (threadnum-1))||(neu_ind == (neurons_in_layer-1))) {
            for (int t=0; t<threads.size(); t++){
                threads.at(t).join();
            }
            
            threads.clear();
            thr_ind = 0;
        }else{
            thr_ind++;
        }
    }
    //=============== ITERATE NEURONS END ===============
    
    //=============== TIME MARK ===============
    if (DEBUG_TIME){
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        
        printf("\n################################\n");
        printf("# LAYER - TOOK %.2f SEC TO PROCESS #\n", elapsed_seconds.count());
        printf("################################\n\n");
    }
}

void Layer::description(){
    printf("[=LAYER %d]\n", tag);
    
    printf("[==OUTPUT LAYER: ");
    for (int i = 0; i<outputLayer.size(); i++){
        printf("(%.2f, i%.2f), ", real(outputLayer.at(i)), imag(outputLayer.at(i)));
    }
    printf("]\n");
    
    for (int i = 0; i<neurons.size(); i++){
        neurons.at(i)->description();
    }
}