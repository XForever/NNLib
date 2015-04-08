#pragma once

#include <vector>
#include <iostream>

using namespace std;

inline double Random()
{
	return (2.0*(double)rand() / RAND_MAX) - 1;
}

class Neuron
{
public:
	int n_Inputs;	//number of inputs
	vector<double> weights;	//weights of inputs
	double activation;	//output value, through Sigmoid function
	double error;	//error value of neuron
	double bias;	//thresold of neuron

	Neuron(int n_Inputs);
	void Initialize();
	bool Calculate(vector<double> input);	//calculate output of given input
	double Sigmoid(double value);	//transfer function
	void PrintWeights();	//print weights of this neuron
};
