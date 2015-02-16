//
//  SOM_ComplexTest.cpp
//  NN_proj
//
//  Created by vsnRain on 15/10/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#include "SOM_ComplexTest.h"

vector<int> cls;
string filename;

void SOM_ComplexTest(){
    
    // ===== CREATING NETWORK =====
    Network *net = new Network();
    net->setTransferType(complexType);
    
    net->addLayer();
    net->layers.back()->addNeurons(3);
    
    
    // ===== CREATING TRAINER =====
    CVSOMTrainer *t = new CVSOMTrainer();
    t->setCycles(100);
    filename = "TrainerData/test3.csv";
    
    #pragma message "TODO: IMPROVE"
    // LEARNING RATE HARDCODED IN TRAINER
    // RADIUS HARDCODED IN TRAINER
    
    // ===== INIT DATA =====
    cvmatrix trainingData;
    prepareData(trainingData);
    
    t->setTrainingData(trainingData);
    // ===== TRAINING =====
    t->train(net);
    
    net->description();
    
    dumpResult(net, trainingData);
    
    // ===== TESTING =====
    for (int i = 0; i<trainingData.size(); i++) {
        net->setInput(trainingData.at(i));
        //net->run();
        
        complex<double> min(MAXFLOAT, MAXFLOAT);
        int winner = -1;
        
        double d[net->layers.at(0)->neurons.size()];
        
        for (int j = 0; j<net->layers.at(0)->neurons.size(); j++) {
            complex<double> distance (0,0);
            
            for (int k = 0; k<net->layers.at(0)->neurons.at(j)->weights.size(); k++) {
                distance += sqr(net->layers.at(0)->neurons.at(j)->weights.at(k) - trainingData.at(i).at(k));
            }
            
            d[j] = magnitude(distance);
            
            if (magnitude(distance) < magnitude(min)){
                min = distance;
                winner = j;
            }
        }
        
        printf("INPUT #%d CLOSEST TO GROUP #%d (SHOULD BE #%d) - ", i, winner, cls.at(i));
        printf("[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n", d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9]);
        
    }
    
    /*
    for (;;) {
        printf("\nINPUT LINE:\n>");
        string s;
        cin >> s;
        
        if (s == "END") {
            break;
        }
        
        net->setInput(trainingData.at(atoi(s.c_str())));
        net->run();
        net->printOutput();
    }
    */
    // ===== CLEANUP =====
    delete t;
    delete net;
}

void prepareData(cvmatrix &trainingData){
    
    //string filename = "TrainerData/complex4.csv";
    cout << "Openning the file "<<filename<<" for reading"<< endl;
    ifstream ifs;
    ifs.open(filename);
    
    if(!ifs){
        cout<<"ERROR - There is no such file as "<<filename<<endl;
        exit(1);
        return;
    }
    
    string line;
    getline(ifs, line);
    
    int i = 0;
    
    trainingData.clear();
    
    while(getline(ifs, line)){
        istringstream s(line);
        
        string t;
        getline(s, t, ',');
        string l;
        getline(s, l, ',');
        string cl;
        getline(s, cl, ',');
        cls.push_back(atoi(cl.c_str()));
        
        string real;
        string imag;
        
        trainingData.push_back(vector<complex<double>> ());
        
        while (getline(s, real,',')&&getline(s, imag, ',')){
            complex<double> val(atof(real.c_str()), atof(imag.c_str()));
            
            trainingData.at(i).push_back(val);
        }
        //cout<<"CYCLE "<<i<<" DONE"<<endl;
        
        i++;
    }
    
    ifs.close();
}

double magnitude(complex<double> arg){
    double res = sqrt(sqr(arg.real()) + sqr(arg.imag()));
    return res;
}

void dumpResult(Network *net, cvmatrix trainingData){
    ofstream ofs;
    string filename = "Out/SOM_Result";
    filename.append(".pbm");
    
    ofs.open (filename);
    
    int x = 186;
    int y = 120;
    
    ofs << "P2" << endl;
    ofs << y << " " << x << endl;
    ofs << net->layers.at(0)->neurons.size()-1 << endl;
    
    int tmp = 0;
    string res = "";
    
    for (int i = 0; i<trainingData.size(); i++) {
        net->setInput(trainingData.at(i));
        
        complex<double> max(MAXFLOAT, MAXFLOAT);
        int winner = -1;
        
        double d[net->layers.at(0)->neurons.size()];
        
        for (int j = 0; j<net->layers.at(0)->neurons.size(); j++) {
            complex<double> distance (0,0);
            
            for (int k = 0; k<net->layers.at(0)->neurons.at(j)->weights.size(); k++) {
                distance += sqr(net->layers.at(0)->neurons.at(j)->weights.at(k) - trainingData.at(i).at(k));
            }
            
            d[j] = magnitude(distance);
            
            if (magnitude(distance) < magnitude(max)){
                max = distance;
                winner = j;
            }
        }
        
        for (int j = 0; j<10; j++){
            res.append(to_string(winner));
            res.append(" ");
        }
        
        if (tmp == 11){
            tmp = 0;
            for (int j = 0; j<1; j++){
                ofs << res << endl;
            }
            res = "";
        }else{
            tmp++;
        }
    }
    
    ofs << endl;
    ofs.close();
}