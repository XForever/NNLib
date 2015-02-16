//
//  Network.cpp
//  NN_proj
//
//  Created by vsnRain on 22/08/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#include "Network.h"

Network::Network(){
    //inputLayer = vector<double> (0);
    //outputLayer = vector<double> (0);
    
    inited = false;
}

Network::~Network(){
    for (int i = 0; i<layers.size(); i++) {
        delete layers.at(i);
    }
    
}

// ========== Getters / Setters ==========

void Network::printOutput(){
    /*
    printf("[==INPUT: ");
    for (int i=0; i<inputLayer.size(); i++){
        printf("%f, ", inputLayer.at(i));
        
    }
    printf(" ]\n");
    */
    
    printf("[==OUTPUT: ");
    for (int i=0; i<outputLayer.size(); i++){
        printf("(%.2f, %.2f), ", outputLayer.at(i).real(), outputLayer.at(i).imag());
        
    }
    printf(" ]\n");
}

vector<complex<double>> Network::getOutput(){
    return outputLayer;
}

// ========== Service Functions ==========

void Network::addLayer(){
    Layer *l = new Layer();
    l->setNetwork(this);
    l->setTag((int)layers.size());
    
    if (layers.size() > 0){
        l->setInputLayer(layers.back()->outputLayer);
    }else{
        l->setInputLayer(inputLayer);
    }
    
    layers.push_back(l);
}

void Network::setInput(vector<complex<double>> inp){
    inputLayer = inp;
    
    if (!layers.empty()) layers.at(0)->setInputLayer(inputLayer);
    
    init();
}

void Network::setInput(vector<double> inp){
    inputLayer.clear();
    
    for (int i=0; i<inp.size(); i++) {
        complex<double> tmp(inp.at(i), 0);
        inputLayer.push_back(tmp);
    }
    
    
    if (!layers.empty()) layers.at(0)->setInputLayer(inputLayer);
    
    init();
}

void Network::setInputFile(string filename){
    inputLayer.clear();
    int sizeX;
    int sizeY;
    
    string path = "TrainerData/Samples/";
    path.append(filename);
    
    cout << "Openning the file "<<path<<" for data extraction"<< endl;
    ifstream ifs;
    ifs.open(path);
    
    
    if(!ifs){
        cout<<"ERROR - There is no such file as "<<path<<endl;
        exit(-1);
    }
    
    getline(ifs, path);
    getline(ifs, path);
    
    
    ifs >> sizeX >> sizeY;
    
    int count = 0;
    while(!ifs.eof() &&  count < sizeX*sizeY){
        int tmp;
        ifs >> tmp;
        inputLayer.push_back(tmp);
        
        count++;
    }
    
    ifs.close();
}

void Network::setTransferType(TransferFunctionType t){
    transferType = t;
}

void Network::init(){
    if (inited) return;
    
    for (int i = 0; i<layers.size(); i++){
        
        if (i==0) {
            layers.at(i)->setInputLayer(inputLayer);
        }else{
            layers.at(i)->setInputLayer(layers.at(i-1)->outputLayer);
        }
        
        layers.at(i)->initNeuronsWeight();
    }
    
    inited = true;
}

void Network::run(){
    for (int i = 0; i<layers.size(); i++){
        
        if (i==0) {
            layers.at(i)->setInputLayer(inputLayer);
        }else{
            layers.at(i)->setInputLayer(layers.at(i-1)->outputLayer);
        }
        
        layers.at(i)->process();
        
        if (i+1 == layers.size()){
            outputLayer = layers.at(i)->outputLayer;
        }
    }
}

void Network::description(){
    printf("[NETWORK: %lu layers]\n", layers.size());
    printf("[==OUTPUT LAYER: ");
    for (int i = 0; i<outputLayer.size(); i++){
        printf("(%.2f, %.2f), ", outputLayer.at(i).real(), outputLayer.at(i).imag());
    }
    printf(" ]\n");
    
    for (int i = 0; i<layers.size(); i++){
        layers.at(i)->description();
    }
}






