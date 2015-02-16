//
//  main.cpp
//  NN_proj
//
//  Created by vsnRain on 22/08/2014.
//  Copyright (c) 2014 vsnrain. All rights reserved.
//

#include "Network.h"
#include "../NNLib_Tests/Tests.h"

int main(int argc, const char * argv[])
{
    cout << "Main started\n";
    
    //RVBP_XORTest();
    //Perceptron_ImageTest();
    SOM_SimpleTest();
    //SOM_ComplexTest();
    
    return 0;
}