//
//  RVSOMTrainer.cpp
//  NN_proj
//
//  Created by vsnRain on 06/10/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#include "RVSOMTrainer.h"

RVSOMTrainer::RVSOMTrainer(){
}

void RVSOMTrainer::setTrainingData(rvmatrix data){
    trainingData = data;
}

void RVSOMTrainer::train(Network *n){
    
    //=============== TIME MARK ===============
    std::chrono::time_point<std::chrono::system_clock> start, end;
    if (DEBUG_TRAINER_TIME){
        start = std::chrono::system_clock::now();
        
        printf("\n#############################\n");
        printf("# TRAINER - TIME MARK DONE  #\n");
        printf("#############################\n\n");
    }
    
    //=============== INIT NETWORK ===============
    network = n;
    network->setInput(trainingData.at(0));
    network->init();
    
    vector<thread> threads;
    
    #pragma message "TODO: POTENTIALLY BAD, CATCH EXCEPTIONS, NOFILE CONTINUE"
    
    //=============== ITERATE LEARN CYCLES BEGIN ===============
    for (int cy_ind = 0; cy_ind<cycles; cy_ind++){
        
        if (DEBUG_TRAINER_CYCLES){
            printf("| CYCLE %d BEGIN\n", cy_ind);
        }
        
        //=============== ITERATE TRAINER DATA BEGIN ===============
        for (int dat_ind = 0; dat_ind<trainingData.size(); dat_ind++){
            
            if (DEBUG_TRAINER_INPUT){
                printf("|=== TRAINING INPUT SET #%d\n", dat_ind);
            }
            
            network->setInput(trainingData.at(dat_ind));
            network->run();
            
            double winnerDistance = MAXFLOAT;
            int winnerIndex = -1;
            int la_ind = 0;
            int thr_ind = 0;
            
            int neurons_in_layer = (int)network->layers.at(la_ind)->neurons.size();
            
            
            //=============== ITERATE NEURONS BEGIN ===============
            for (int neu_ind=0; neu_ind<neurons_in_layer; neu_ind++){
                
                pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
                
                //=============== LEARNING THREAD BEGIN ===============
                threads.push_back(thread([=, &winnerIndex, &winnerDistance, &mut]() -> void {
                    
                    if (DEBUG_TRAINER_THREADING){
                        printf("|=== |=== |=== LEARNING THREAD #%d STARTED FOR NEURON #%d\n", thr_ind, neu_ind);
                    }
                    
                    double distance = 0;
                    
                    for (int wei_ind=0; wei_ind<network->layers.at(la_ind)->neurons.at(neu_ind)->weights.size(); wei_ind++){
                        distance += sqr(network->layers.at(la_ind)->neurons.at(neu_ind)->weights.at(wei_ind).real()-trainingData.at(dat_ind).at(wei_ind));
                    }
                    
                    pthread_mutex_lock(&mut);
                    if (distance < winnerDistance){
                        winnerDistance = distance;
                        winnerIndex = neu_ind;
                    }
                    pthread_mutex_unlock(&mut);
                    
                    if (DEBUG_TRAINER_NEURON){
                        char str[150];
                        
                        sprintf(str, "|=== |=== |=== |=== LAYER #%d ", la_ind);
                        sprintf(str, "%s| NEURON: %d ", str, neu_ind);
                        sprintf(str, "%s| DISTANCE: %f\n", str, distance);
                        
                        printf("%s", str);
                    }
                    
                    if (DEBUG_TRAINER_THREADING){
                        printf("|=== |=== |=== THREAD %d DONE\n", thr_ind);
                    }
                }));
                //=============== LEARNING THREAD END ===============
                
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
            
            if (DEBUG_TRAINER_INPUT){
                printf("|=== |=== NEURON %d IS A WINNER AT INPUT %d\n", winnerIndex, dat_ind);
            }
            
            updateWeights(winnerIndex, dat_ind, cy_ind);
        }
        //=============== ITERATE TRAINER DATA END ===============
        
        if (DEBUG_TRAINER_CYCLES){
            printf("| CYCLE %d END\n", cy_ind);
        }
    }
    //=============== ITERATE LEARN CYCLES END ===============
    
    // =============== TIME MARK ===============
    if (DEBUG_TRAINER_TIME){
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        
        printf("\n#############################\n");
        printf("# TRAINING TOOK %.2f SEC #\n", elapsed_seconds.count());
        printf("#############################\n\n");
    }
}

void RVSOMTrainer::updateSingleWeight(int nodeNumber, int signalNumber, int iter){
    for (int i=0; i<network->layers.at(0)->neurons.at(nodeNumber)->weights.size(); i++){
        
        double diff = network->layers.at(0)->neurons.at(nodeNumber)->inputLayer.at(i).real() - network->layers.at(0)->neurons.at(nodeNumber)->weights.at(i).real();
        network->layers.at(0)->neurons.at(nodeNumber)->weights.at(i) += learningRate(iter) * diff;
    }
}

void RVSOMTrainer::updateWeights(int nodeNumber, int signalNumber, int iter){
    
    updateSingleWeight(nodeNumber, signalNumber, iter);
    
    
    for (int i=1; i<=radius(iter); i++){
        if ((nodeNumber-i) >= 0) updateSingleWeight(nodeNumber-i, signalNumber, iter);
        if ((nodeNumber+i) < network->layers.at(0)->neurons.size()) updateSingleWeight(nodeNumber+i, signalNumber, iter);
    }
    
    if (DEBUG_TRAINER_NEURON){
        char str[150];
        
        sprintf(str, "|=== |=== |=== |=== ITER #%d ", iter);
        sprintf(str, "%s| RATE: %f ", str, learningRate(iter));
        sprintf(str, "%s| RADIUS: %d \n", str, radius(iter));
        
        for (int i=0; i < network->layers.at(0)->neurons.size(); i++){
            if (i == nodeNumber) printf("=>");
            network->layers.at(0)->neurons.at(i)->description();
        }
        
        printf("%s", str);
    }
}

double RVSOMTrainer::learningRate(int iter){
    if (iter <= cycles*0.4) return 0.8;
    else if (iter > cycles*0.4 && iter <= cycles*0.6) return 0.3;
    else if (iter > cycles*0.6) return 0.15;
    else return 0.15;
}

int RVSOMTrainer::radius(int iter){
    ///*
    if (iter <= cycles*0.4) return 3;
    else
    if (iter <= cycles*0.6) return 2;
    else
    if (iter <= cycles*0.3) return 1;
    else return 0;
    //*/
     
    return 0;
}