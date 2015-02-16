//
//  Neuron.cpp
//  NN_proj
//
//  Created by vsnRain on 22/08/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#include "Neuron.h"

Neuron::Neuron(){
    inputSize = 0;
    
    tag = (int)arc4random();
}

Neuron::~Neuron(){
    
}

void Neuron::initWeights(){
    for (int i=0; i<inputLayer.size(); i++){
        
        double rand_r = (double)arc4random()/ARC4RANDOM_MAX;
        double rand_i = (double)arc4random()/ARC4RANDOM_MAX;
        
        complex<double> t (rand_r, 0);
        
        switch (layer->network->transferType) {
            case complexType:
                //t = complex<double> (0.0, 0.0);
                t = complex<double> ((rand_r*2)-1, (rand_i*2)-1);
                break;
            case stepTransferType:
                break;
            case sigmoidTransferType:
                break;
            case bipolarSigmoidTransferType:
                t = complex<double> ((rand_r*2)-1, (rand_i*2)-1);
                break;
            case tanhTransferType:
                break;
            default:
                printf("NEURON: WRONG TRANSFER TYPE\n");
                break;
        }
        weights.push_back(t);
    }
    
    if(layer->network->transferType == complexType){
        double rand_r = (double)arc4random()/ARC4RANDOM_MAX;
        double rand_i = (double)arc4random()/ARC4RANDOM_MAX;
        bias_weight = complex<double> (rand_r, rand_i);
    }else if (layer->network->transferType == stepTransferType) {
        bias_weight = 1;
    }else{
        double rand_r = (double)arc4random()/ARC4RANDOM_MAX;
        bias_weight = complex<double> (rand_r, 0);
    }
}

void Neuron::setLayer(Layer *l){
    layer = l;
}

void Neuron::setInputLayer(vector<complex<double>> inp){
    inputLayer = inp;
}

complex<double> Neuron::getOutput(){
    return output;
}

void Neuron::check(){
    if (inputLayer.size() != weights.size()){
        printf("NEURON: SIZE MISMATCH\n");
        exit(1);
    }
}

void Neuron::setTag(int t){
    tag = t;
}

complex<double> Neuron::count(){
    check();
    
    complex<double> sum = 0;
    
    for (int i=0; i<inputLayer.size(); i++){
        sum += weights.at(i) * inputLayer.at(i);
    }
    
    sum += layer->bias * bias_weight;
    
    return sum;
}

complex<double> Neuron::activate(){
    complex<double> res = count();
    complex<double> ret = 0;
    
    //Radial Basis NN
    
    switch (layer->network->transferType) {
        case stepTransferType:
            (res.real() < layer->bias.real()) ? ret = 0 : ret = 1;
            break;
        case sigmoidTransferType:
            ret = sigmoidFunction(res + layer->bias);
            break;
        case bipolarSigmoidTransferType:
            ret = bipolarSigmoidFunction(res + layer->bias);
            break;
        case tanhTransferType:
            ret = tanhFunction(res + layer->bias);
            break;
        case complexType:
            ret = tanhFunction(res + layer->bias);
            break;
        default:
            printf("NEURON: WRONG TRANSFER TYPE\n");
            exit(1);
            ret = 0;
            break;
    }
    
    return ret;
}

void Neuron::printWeights(){
    printf("[");
    for (int i=0; i<weights.size(); i++){
        printf("(%.2f, i%.2f), ", real(weights.at(i)), imag(weights.at(i)));
    }
    printf("]\n");
}

void Neuron::dumpWeightsMatrix(){
    
    int x = sqrt(weights.size());
    if (x<0){
        printf("NEURON - CAN'T DUMP WEIGHTS");
        return;
    }
    
    ofstream ofs;
    string filename = "Out/L";
    filename.append(to_string(layer->tag));
    filename.append("/Neuron");
    filename.append(to_string(tag));
    filename.append(".pbm");
    
    ofs.open (filename);
    
    ofs << "P2" << endl;
    ofs << x << " " << x << endl;
    ofs << 6 << endl;
    
    for (int i=0; i<x; i++){
        for (int j=0; j<x; j++){
            ofs << ceil(real(weights.at(i*x+j))+3)<<" ";
        }
        ofs << endl;
    }
    ofs << endl;
    
    ofs.close();
}

void Neuron::dumpWeightsArray(){
    ofstream ofs;
    string filename = "Out/L";
    filename.append(to_string(layer->tag));
    filename.append("/Neuron");
    filename.append(to_string(tag));
    filename.append(".txt");
    
    ofs.open (filename);
    
    ofs << weights.size() << " " << endl;
    
    for (int i=0; i<weights.size(); i++){
        ofs << weights.at(i)<<" ";
    }
    ofs << endl;
    
    ofs.close();
}

complex<double> Neuron::sigmoidFunction(complex<double> val){
    complex<double> ret = 1.0 / (1.0 + exp(-val));
    return ret;
}

complex<double> Neuron::bipolarSigmoidFunction(complex<double> val){
    #pragma message "TODO: NO COMPLEX"
    complex<double> ret = (2.0 / (1.0 + exp(-val))).real() - 1;
    return ret;
}

complex<double> Neuron::tanhFunction(complex<double> val){
    complex<double> ret = (exp(val) - exp(-val))/(exp(val) + exp(-val));
    return ret;
}

double Neuron::magnitude(complex<double> comp){
    double ret = sqrt(sqr(comp.real())+sqr(comp.imag()));
    return ret;
}

void Neuron::description(){
    printf("[==NEURON: ");
    
    for (int i = 0; i<weights.size(); i++){
        printf("(%.2f, i%.2f), ", real(weights.at(i)), imag(weights.at(i)));
    }
    printf("]\n");
}