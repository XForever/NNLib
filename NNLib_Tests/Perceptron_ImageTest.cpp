//
//  ImageTest.cpp
//  NN_proj
//
//  Created by vsnRain on 03/10/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#include "Perceptron_ImageTest.h"

void Perceptron_ImageTest(){
    
    // ===== CREATING NETWORK =====
    Network *net = new Network();
    net->setTransferType(stepTransferType);
    
    net->addLayer();
    net->layers.back()->addNeurons(14);
    
    
    // ===== CREATING TRAINER =====
    RVBPTrainer *t = new RVBPTrainer();
    t->setCycles(10);
    t->setLearningRate(0.8);
    
    // ===== INIT DATA =====
    rvmatrix trainingData;
    vector<vector<int>> targetNeurons;
    
    prepareData(trainingData, targetNeurons);
    
    t->setTrainingData(trainingData);
    t->setTargetNeurons(targetNeurons);
    
    // ===== TRAINING =====
    t->train(net);
    
    
    // ===== TESTING =====
    for (;;) {
        printf("\nINPUT FILE:\n>");
        string s;
        cin >> s;
        
        if (s == "END") {
            break;
        }
        
        net->setInputFile(s);
        net->run();
        net->printOutput();
    }
    
    // ===== CLEANUP =====
    delete t;
    delete net;
}

void prepareData(rvmatrix &trainingData, vector<vector<int>> &targetNeurons){
    vector<string> targetValues;
    
    string filename = "TrainerData/list.txt";
    cout << "Openning the file "<<filename<<" for reading"<< endl;
    ifstream ifs;
    ifs.open(filename);
    
    if(!ifs){
        cout<<"ERROR - There is no such file as "<<filename<<endl;
        return;
    }
    
    string file;
    string type;
    set<string> values;
    
    getline(ifs, file);
    
    while(!ifs.eof())
    {
        ifs >> file >> type;
        
        file = "TrainerData/Samples/" + file;
        vector<double> data = extractData(file);
        trainingData.push_back(data);
        
        values.insert(type);
        targetValues.push_back(type);
    }
    
    if (trainingData.size() != targetValues.size()){
        printf("TRAINER - ERROR - DATA<->NEURONS MISMATCH\n");
        exit(1);
    }
    
    for(auto i : values) {
        vector<int> tmp;
        
        for (int j=0; j<targetValues.size(); j++){
            if (targetValues.at(j) == i) tmp.push_back(j);
        }
        
        if (!tmp.empty()) targetNeurons.push_back(tmp);
    }
    
    ifs.close();
}

vector<double> extractData(string file){
    
    vector<double> result;
    
    cout << "Openning the file "<<file<<" for data extraction"<< endl;
    ifstream ifs;
    ifs.open(file);
    
    
    if(!ifs){
        cout<<"ERROR - There is no such file as "<<file<<endl;
        return result;
    }
    
    getline(ifs, file);
    getline(ifs, file);
    
    #pragma message "TODO: POTENTIALLY BAD"
    int sizeX, sizeY;
    ifs >> sizeX >> sizeY;
    
    int count = 0;
    while(!ifs.eof() &&  count < sizeX*sizeY){
        int tmp;
        ifs >> tmp;
        result.push_back(tmp);
        
        count++;
    }
    
    ifs.close();
    
    return result;
}