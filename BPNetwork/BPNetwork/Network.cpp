#include "Network.h"
#include <time.h>

using namespace std;

Network::Network(int n_Inputs, int n_Hiddens, int n_Outputs)
{
	this->n_Inputs = n_Inputs;
	this->n_Hiddens = n_Hiddens;
	this->n_Outputs = n_Outputs;
}

void Network::SetNeuronNumbersPerLayer(int n_Inputs, int n_Hiddens, int n_Outputs)
{
	this->n_Inputs = n_Inputs;
	this->n_Hiddens = n_Hiddens;
	this->n_Outputs = n_Outputs;
}

void Network::SetLearningRate(double lr_w_HiddenLayer, double lr_w_OutputLayer, double lr_b_HiddenLayer, double lr_b_OutputLayer)
{
	this->lr_w_HiddenLayer = lr_w_HiddenLayer;
	this->lr_w_OutputLayer = lr_w_OutputLayer;
	this->lr_b_HiddenLayer = lr_b_HiddenLayer;
	this->lr_b_OutputLayer = lr_b_OutputLayer;
}

void Network::SetCycles(int cycles)
{
	this->cycles = cycles;
}

void Network::CreateNetwork()
{
	hiddenLayer = new Layer(n_Hiddens, n_Inputs);
	outputLayer = new Layer(n_Outputs, n_Hiddens);
}

void Network::Initialize()
{
	srand((unsigned)time(NULL));

	/*initialize the weights of hidden layer*/
	hiddenLayer->Initialize();
	/*initialize the weights of output layer*/
	outputLayer->Initialize();
}

bool Network::Train(matrix inputs, matrix outputs)
{
	vector<double> error_OutputLayer(n_Outputs);	//希望输出与实际输出的偏差
	vector<double> error_HiddenLayer(n_Hiddens);	//隐含层的校正误差

	for (int i = 0; i < cycles; i++)
	{
		errorSum = 0;
		for (int j = 0; j < inputs.size(); j++)
		{
			vector<double> output;

			if (!Calculate(inputs[j], output))
				return false;

			/*for(int l=0;l<output.size();l++)
				cout<<output[l]<<" ";*/

			/*update the weights of output layer*/
			for (int k = 0; k < n_Outputs; k++)
			{
				/*compute error term*/
				error_OutputLayer[k] = (outputs[j][k] - output[k])*output[k] * (1 - output[k]);
				/*update the sum of error*/
				errorSum += (outputs[j][k] - output[k])*(outputs[j][k] - output[k]);

				/*update the weights of neurons*/
				for (int m = 0; m < n_Hiddens; m++)
					outputLayer->neurons[k]->weights[m] += lr_w_OutputLayer*error_OutputLayer[k] * hiddenLayer->neurons[m]->activation;
			}

			/*update the weights of hidden layer*/
			for (int m = 0; m < n_Hiddens; m++)
			{
				error_HiddenLayer[m] = 0;
				for (int k = 0; k < n_Outputs; k++)
					error_HiddenLayer[m] += error_OutputLayer[k] * outputLayer->neurons[k]->weights[m];
				error_HiddenLayer[m] *= hiddenLayer->neurons[m]->activation*(1 - hiddenLayer->neurons[m]->activation);

				/*update the weights of neurons*/
				for (int n = 0; n < n_Inputs; n++)
					hiddenLayer->neurons[m]->weights[n] += lr_w_HiddenLayer*error_HiddenLayer[m] * inputs[j][n];
			}

			/*update the bias of output layer*/
			for (int k = 0; k < n_Outputs; k++)
				outputLayer->neurons[k]->bias += lr_b_OutputLayer*error_OutputLayer[k];

			/*update the bias of hidden layer*/
			for (int m = 0; m < n_Hiddens; m++)
				hiddenLayer->neurons[m]->bias += lr_b_HiddenLayer*error_HiddenLayer[m];
		}
		//cout<<endl;
	}

	return true;
}

bool Network::Calculate(vector<double> input, vector<double>& output)
{
	vector<double> output_HiddenLayer;

	if (!hiddenLayer->Calculate(input, output_HiddenLayer))
		return false;
	if (!outputLayer->Calculate(output_HiddenLayer, output))
		return false;

	return true;
}

void Network::PrintWeights()
{
	cout << "HiddenLayer:" << endl;

	for (int i = 0; i < n_Hiddens; i++)
	{
		for (int j = 0; j < n_Inputs; j++)
			cout << hiddenLayer->neurons[i]->weights[j] << " ";
		cout << "bias:" << hiddenLayer->neurons[i]->bias << endl;
	}

	cout << "OutputLayer:" << endl;

	for (int i = 0; i < n_Outputs; i++)
	{
		for (int j = 0; j < n_Hiddens; j++)
			cout << outputLayer->neurons[i]->weights[j] << " ";
		cout << "bias:" << outputLayer->neurons[i]->bias << endl;
	}
}