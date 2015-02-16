//
//  Trainer.cpp
//  NN_proj
//
//  Created by vsnRain on 25/08/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#include "Trainer.h"

Trainer::Trainer(){
    setCycles(1);
    setLearningRate(0.5);
}

void Trainer::setCycles(int c){
    cycles = c;
}

void Trainer::setLearningRate(double r){
    learningRate = r;
}