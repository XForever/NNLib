//
//  CVSOMTrainer.cpp
//  NN_proj
//
//  Created by vsnRain on 15/10/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#include "CVSOMTrainer.h"

CVSOMTrainer::CVSOMTrainer(){
}

void CVSOMTrainer::setTrainingData(cvmatrix data){
    trainingData = data;
}

void CVSOMTrainer::train(Network *n){
    
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
            //network->run();
        
            int la_ind = 0;
            int neurons_in_layer = (int)network->layers.at(la_ind)->neurons.size();
            
            complex<double> winnerDistance (MAXFLOAT, MAXFLOAT);
            int winnerIndex = -1;
            
            threads.clear();
            int thr_ind = 0;
            
            pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
            
            //=============== ITERATE NEURONS BEGIN ===============
            for (int neu_ind=0; neu_ind<neurons_in_layer; neu_ind++){
                
                //=============== LEARNING THREAD BEGIN ===============
                threads.push_back(thread([=, &winnerIndex, &winnerDistance, &mut]() -> void {
                    
                    if (DEBUG_TRAINER_THREADING){
                        printf("|=== |=== |=== LEARNING THREAD #%d STARTED FOR NEURON #%d\n", thr_ind, neu_ind);
                    }
                    
                    complex<double> distance (0,0);
                    complex<double> norm1 (0,0);
                    complex<double> norm2 (0,0);
                    ///*
                    for (int wei_ind=0; wei_ind<network->layers.at(la_ind)->neurons.at(neu_ind)->weights.size(); wei_ind++){
                        distance += sqr(network->layers.at(la_ind)->neurons.at(neu_ind)->weights.at(wei_ind) - trainingData.at(dat_ind).at(wei_ind));
                    }
                    //*/
                    
                    /*
                    for (int wei_ind=0; wei_ind<network->layers.at(la_ind)->neurons.at(neu_ind)->weights.size(); wei_ind++){
                        distance += network->layers.at(la_ind)->neurons.at(neu_ind)->weights.at(wei_ind) * trainingData.at(dat_ind).at(wei_ind);
                        
                        norm1 += sqr(trainingData.at(dat_ind).at(wei_ind));
                        norm2 += sqr(network->layers.at(la_ind)->neurons.at(neu_ind)->weights.at(wei_ind));
                    }
                    
                    if ((norm1 == complex<double>(0,0))||(norm2 == complex<double>(0,0))) distance = complex<double> (0,0);
                    else distance = distance/(sqrt(norm1)*sqrt(norm2));
                    */
                    
                    pthread_mutex_lock(&mut);
                    if (magnitude(distance) < magnitude(winnerDistance)){
                        winnerDistance = distance;
                        winnerIndex = neu_ind;
                    }
                    pthread_mutex_unlock(&mut);
                    
                    if (DEBUG_TRAINER_NEURON){
                        char str[150];
                        
                        sprintf(str, "|=== |=== |=== |=== LAYER #%d ", la_ind);
                        sprintf(str, "%s| NEURON: %d ", str, neu_ind);
                        sprintf(str, "%s| DISTANCE: (%.2f, %.2f)\n", str, distance.real(), distance.imag());
                        
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
                printf("|=== |=== NEURON %d IS A WINNER AT INPUT %d WITH DISTANCE %f\n", winnerIndex, dat_ind, magnitude(winnerDistance));
            }
            
            updateWeights(winnerIndex, cy_ind);
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

void CVSOMTrainer::updateSingleWeight(int nodeNumber, int iter, bool weak){
    for (int i=0; i<network->layers.at(0)->neurons.at(nodeNumber)->weights.size(); i++){
        
        complex<double> update = network->layers.at(0)->neurons.at(nodeNumber)->inputLayer.at(i) - network->layers.at(0)->neurons.at(nodeNumber)->weights.at(i);
        
        if (DEBUG_TRAINER_SOM_UPDATE_DETAIL){
            printf("== REAL DIFF WAS (%f, %f)\n", update.real(), update.imag());
            printf("== REAL DIFF MAGNITUDE WAS %f\n", magnitude(update));
            
            printf("== LRATE IS %f\n", learningRate(iter));
        }
        
        update *= learningRate(iter);
        if (weak) update *= 0.005;
        
        network->layers.at(0)->neurons.at(nodeNumber)->weights.at(i) += update;
        
        if (DEBUG_TRAINER_SOM_UPDATE_DETAIL){
            printf("== DIFF*RATE WAS (%f, %f)\n", update.real(), update.imag());
            printf("== DIFF*RATE MAGNITUDE WAS %f\n", magnitude(update));
        }
    }
}

void CVSOMTrainer::updateWeights(int nodeNumber, int iter){
    
    updateSingleWeight(nodeNumber, iter, false);
    
    for (int i=1; i<=radius(iter); i++){
        if ((nodeNumber-i) >= 0) updateSingleWeight(nodeNumber-i, iter, true);
        if ((nodeNumber+i) < network->layers.at(0)->neurons.size()) updateSingleWeight(nodeNumber+i, iter, true);
    }
    
    if (DEBUG_TRAINER_SOM_UPDATE){
        char str[150];
        
        sprintf(str, "|=== |=== |=== |=== ITER #%d ", iter);
        sprintf(str, "%s| RATE: %f ", str, learningRate(iter));
        sprintf(str, "%s| RADIUS: %d \n", str, radius(iter));
        
        for (int i=0; i < network->layers.at(0)->neurons.size(); i++){
            if (i == nodeNumber) printf("UPDATE MARK => ");
            network->layers.at(0)->neurons.at(i)->description();
        }
        
        printf("%s", str);
    }
}

double CVSOMTrainer::learningRate(int iter){
    
    if (iter <= cycles*0.25) return 0.9;
    else if (iter > cycles*0.25 && iter <= cycles*0.4) return 0.6;
    else if (iter > cycles*0.5 && iter <= cycles*0.75) return 0.15;
    else
    
    
    return 0.01;
    //*/
}

int CVSOMTrainer::radius(int iter){
    if (iter <= cycles*0.7) return 1;
    else return 0;
}

double CVSOMTrainer::magnitude(complex<double> arg){
    double res = sqrt(sqr(arg.real()) + sqr(arg.imag()));
    return  res;
}