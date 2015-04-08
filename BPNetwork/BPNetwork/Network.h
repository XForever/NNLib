#include <vector>
#include "Neuron.h"
#include "Layer.h"

#define matrix vector< vector< double > >

using namespace std;

class Network
{
public:
	int n_Inputs;	//number of inputs
	int n_Hiddens;	//number of neurons of hidden layer
	int n_Outputs;	//number of neurons of output layer

	double lr_w_HiddenLayer;	//learning rate of weights of hidden layer
	double lr_w_OutputLayer;	//learning rate of weights of output layer
	double lr_b_HiddenLayer;	//learning rate of bias of hidden layer
	double lr_b_OutputLayer;	//learning rate of bias of output layer

	int cycles;	//times of training
	double errorSum;	//sum of error of one-time training 
	double errorAllowed;

	Layer *hiddenLayer;
	Layer *outputLayer;

	Network(int n_Inputs, int n_Hiddens, int n_Outputs);
	void SetNeuronNumbersPerLayer(int n_Inputs, int n_Hiddens, int n_Outputs);	//set numbers of neurons of each layer
	void SetLearningRate(double lr_w_HiddenLayer, double lr_w_OutputLayer, double lr_b_HiddenLayer, double lr_b_OutputLayer);	//set learning rate of weights and bias of hidden layer and output layer
	void SetCycles(int cycles);	//set times of training
	void CreateNetwork();	//create this network
	void Initialize();	//initialize weights and bias for each neuron and layer
	bool Train(matrix input, matrix output);	//train this network with given inputs and target outputs
	bool Calculate(vector<double> inputs, vector<double>& outputs);	//calculate output of given input
	void PrintWeights();	//print weigths of each neuron
};

