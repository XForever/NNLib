#include "Neuron.h"

class Layer
{
public:
	int n_Neurons;	//number of neurons
	int n_InputsPerNeuron;	//number of inputs for each neuron in this layer
	vector<Neuron *> neurons;	//vector of neurons

	Layer(int n_Neurons, int n_InputsPerNeuron);
	void Initialize();
	bool Calculate(vector<double> input, vector<double>& output);	//calculate output of given input
};