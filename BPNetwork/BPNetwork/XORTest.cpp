#include "Network.h"
#include <iostream>

using namespace std;

void main()
{
	Network *net = new Network(2, 8, 1);//inputs:2 hidden neurons:8 output neuron:1

	net->CreateNetwork();
	net->Initialize();
	net->SetLearningRate(0.5, 0.5, 0.5, 0.5);//learning rate of weights and bias of hidden layer and output layer
	net->SetCycles(1000);	//set times of training 
	//net->PrintWeights();

	matrix trainingData = {	//training data
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 }
	};

	matrix targetOutput = {	//target output
		{ 0 },
		{ 1 },
		{ 1 },
		{ 0 }
	};

	cout << "---------------------" << endl;
	if (net->Train(trainingData, targetOutput))
		cout << "Trained Successfully!" << endl;
	cout << "---------------------" << endl;

	matrix output(4);

	for (int i = 0; i < trainingData.size(); i++)
	{
		if (net->Calculate(trainingData[i], output[i]))
			cout << trainingData[i][0] << " ¨’ " << trainingData[i][1] << " = " << output[i][0];
		cout << endl;
	}

	//net->PrintWeights();

	system("pause");
}