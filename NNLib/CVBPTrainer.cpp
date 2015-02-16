//
//  CVBPTrainer.cpp
//  NN_proj
//
//  Created by vsnRain on 14/10/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

/*

#include "CVBPTrainer.h"

CVBPTrainer::CVBPTrainer(){
}

void CVBPTrainer::setTargetNeurons(vector<vector<int>> target){
    targetNeurons = target;
}

void CVBPTrainer::setTrainingData(cvmatrix data){
    trainingData = data;
}

void CVBPTrainer::train(Network *n){
    
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
    network->setInputComplex(trainingData.at(0));
    network->init();
    
    vector<thread> threads;
    
#pragma message "TODO: POTENTIALLY BAD, CATCH EXCEPTIONS, NOFILE CONTINUE"
    //=============== CHECK EXCEPTIONS ===============
    if (targetNeurons.empty()){
        printf("TRAINER - ERROR - DON'T NEED TO TRAIN ANYTHING\n");
        exit(1);
    }
    reqNeurons = (int)targetNeurons.size();
    if (network->layers.back()->neurons.size() < reqNeurons){
        printf("TRAINER - ERROR - NOT ENOUGH NEURONS TO TRAIN\n");
        exit(1);
    }
    
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
            
            network->setInputComplex(trainingData.at(dat_ind));
            network->run();
            
            //=============== BACKITERATE LAYERS ===============
            for(int la_ind = (int)network->layers.size()-1; la_ind>=0; la_ind--){
                if (DEBUG_TRAINER_LAYERS){
                    printf("|=== |=== PROCESSING LAYER #%d\n", la_ind);
                }
                
                int neurons_in_layer = (int)network->layers.at(la_ind)->neurons.size();
                int thr_ind = 0;
                
                //=============== ITERATE NEURONS BEGIN ===============
                for (int neu_ind=0; neu_ind<neurons_in_layer; neu_ind++){
                    
                    //=============== LEARNING THREAD BEGIN ===============
                    threads.push_back(thread([=]() -> void {
                        
                        if (DEBUG_TRAINER_THREADING){
                            printf("|=== |=== |=== LEARNING THREAD #%d STARTED FOR NEURON #%d\n", thr_ind, neu_ind);
                        }
                        
                        complex<double> result = network->layers.at(la_ind)->neurons.at(neu_ind)->activate().real();
                        complex<double> count = network->layers.at(la_ind)->neurons.at(neu_ind)->count().real();
                        
                        int target = -1;
                        complex<double> err = 0;
                        
                        if (la_ind == (network->layers.size()-1)){
                            target = 0;
                            
                            for (int n = 0; n< targetNeurons.at(neu_ind).size(); n++){
                                if (targetNeurons.at(neu_ind).at(n) == dat_ind){
                                    target = 1;
                                    break;
                                }
                            }
                            
                            err = (target - result) * errorDerivative(result);
                            
                        }else{
                            err = 0;
                            
                            for (int t = 0; t<network->layers.at(la_ind+1)->neurons.size(); t++) {
                                err += network->layers.at(la_ind+1)->neurons.at(t)->error.real() * network->layers.at(la_ind+1)->neurons.at(t)->weights.at(neu_ind).real();
                            }
                            
                            err = err * errorDerivative(result);
                        }
                        
                        network->layers.at(la_ind)->neurons.at(neu_ind)->error = err;
                        
                        if (DEBUG_TRAINER_NEURON){
                            char str[150];
                            
                            sprintf(str, "|=== |=== |=== |=== LAYER #%d ", la_ind);
                            sprintf(str, "%s| NEURON: %d ", str, neu_ind);
                            sprintf(str, "%s| COUNT: %.2f ", str, count);
                            sprintf(str, "%s| RESULT: %.2f ", str, result);
                            sprintf(str, "%s| BIAS_W: %.2f ", str, network->layers.at(la_ind)->neurons.at(neu_ind)->bias_weight.real());
                            sprintf(str, "%s| TARGET: %d ", str, target);
                            sprintf(str, "%s| ERR: %f\n", str, err);
                            
                            printf("%s", str);
                        }
                        
                        
                        for (int m = 0; m<network->layers.at(la_ind)->neurons.at(neu_ind)->inputLayer.size(); m++){
                            
                            complex<double> weight_delta = learningRate * err * network->layers.at(la_ind)->neurons.at(neu_ind)->inputLayer.at(m).real();
                            
                            network->layers.at(la_ind)->neurons.at(neu_ind)->weights.at(m) += weight_delta;
                        }
                        
                        if (network->transferType != stepTransferType) network->layers.at(la_ind)->neurons.at(neu_ind)->bias_weight += learningRate * err;
                        
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
                
                if (DEBUG_TRAINER_LAYERS){
                    //for (int t=0; t<network->layers.at(la_ind)->neurons.size(); t++) network->layers.at(la_ind)->neurons.at(t)->printWeights();
                    printf("|=== |=== LAYER #%d DONE, PROPAGATE LOWER\n", la_ind);
                }
            }
            //=============== BACKITERATE LAYERS DONE ===============
            
            if (DEBUG_TRAINER_INPUT){
                printf("|=== TRAINING INPUT SET #%d DONE\n", dat_ind);
            }
        }
        //=============== ITERATE TRAINER DATA END ===============
        
        if (DEBUG_TRAINER_CYCLES){
             for (int j=0; j<network->layers.at(0)->neurons.size(); j++){
             network->layers.at(0)->neurons.at(j)->dumpWeights(sizeX, sizeY);
             //network->layers.at(0)->neurons.at(i)->printWeights();
             }
             
            
            printf("| CYCLE %d END\n", cy_ind);
        }
    }
    //=============== ITERATE LEARN CYCLES END ===============
    
    if (DEBUG_TRAINER_DUMP){
        for (int i=0; i<network->layers.size(); i++){
            for (int j=0; j<network->layers.at(0)->neurons.size(); j++){
                network->layers.at(0)->neurons.at(j)->dumpWeightsMatrix();
                //network->layers.at(0)->neurons.at(i)->printWeights();
            }
        }
        
        printf("WEIGHTS DUMPED\n");
    }
    
    // =============== TIME MARK ===============
    if (DEBUG_TRAINER_TIME){
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        
        printf("\n#############################\n");
        printf("# TRAINING TOOK %.2f SEC #\n", elapsed_seconds.count());
        printf("#############################\n\n");
    }
}

complex<double> CVBPTrainer::errorDerivative(complex<double> val){
    double ret = 0;
    
    switch (network->transferType) {
        case stepTransferType:
            ret = 1;
            break;
        case sigmoidTransferType:
            ret = (1-val)*val;
            break;
        case bipolarSigmoidTransferType:
            ret = 0.5*(1-val)*(1+val);
            break;
        case tanhTransferType:
            ret = 1 - sqr(val);
            break;
        default:
            printf("NEURON: WRONG TRANSFER TYPE\n");
            exit(0);
            break;
    }
    
    return ret;
}

*/