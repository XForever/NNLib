#include "Neuron.h"
#include <math.h>

using namespace std;

Neuron::Neuron(int n_Inputs)
{
	this->n_Inputs = n_Inputs;
	weights.reserve(n_Inputs);
}

void Neuron::Initialize()
{
	for (int i = 0; i < n_Inputs; i++)
		weights.push_back(Random());
	bias = Random();
}

bool Neuron::Calculate(vector<double> input)
{
	if (input.size() != n_Inputs)	//input feature vector's dimension is not equals to number of inputs of network
		return false;

	double inputSum = 0;	//sum of multiply inputs by weights

	for (int i = 0; i < n_Inputs; i++)
		inputSum += weights[i] * input[i];
	/*plus bias term*/
	inputSum += bias;
	/*compute sigmoid fuction's output*/
	activation = Sigmoid(inputSum);

	return true;
}

double Neuron::Sigmoid(double value)
{
	double response = 1.0;  //control steep degree of sigmoid function

	return (1 / (1 + exp(-value / response)));
}

void Neuron::PrintWeights()
{
	for (int i = 0; i < n_Inputs; i++)
		cout << weights[i] << " ";
	cout << "bias:" << bias << endl;
}