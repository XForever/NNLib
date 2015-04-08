#include "Layer.h"

using namespace std;

Layer::Layer(int n_Neurons, int n_InputsPerNeuron)
{
	this->n_Neurons = n_Neurons;
	this->n_InputsPerNeuron = n_InputsPerNeuron;

	for (int i = 0; i < n_Neurons; i++)
		neurons.push_back(new Neuron(n_InputsPerNeuron));
}

void Layer::Initialize()
{
	for (int i = 0; i < n_Neurons; i++)
		neurons[i]->Initialize();
}

bool Layer::Calculate(vector<double> input, vector<double>& output)
{
	for (int i = 0; i < n_Neurons; i++)
	{
		if (!neurons[i]->Calculate(input))
			return false;
		output.push_back(neurons[i]->activation);
	}

	return true;
}