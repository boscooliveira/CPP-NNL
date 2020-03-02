#include "PerceptronNetwork.h"
#include "Neuron.h"
#include <cstdlib>
#include <cstdio>

float NNL::Perceptron::SLP::PerceptronNetwork::GetResult(float* inputs) const
{
	return MainNeuron->ProcessInputs(inputs);
}

void NNL::Perceptron::SLP::PerceptronNetwork::Train(int trainSetQtt, float* trainInputs, float* trainResults)
{
	int iteraction = 0;
	float sumErrors = 0;
	do
	{		
		sumErrors = 0;
		for (int i = 0; i < trainSetQtt; i++)
		{
			sumErrors += MainNeuron->Train(&(trainInputs[i* NumInputs]), trainResults[i]);
		}
		std::printf("error : {%3.3f}", sumErrors);
	} while (++iteraction < MaxIteractions && sumErrors > MinQuadraticError);
}

NNL::Perceptron::SLP::PerceptronNetwork::PerceptronNetwork(int numInputs, float learningRate, int maxIteractions, float minQuadraticError)
{
	float *randomWeights = new float[numInputs];
	for (int i = 0; i < numInputs; i++)
	{
		randomWeights[i] = (rand() % 100) / 100.0f;
	}
	MainNeuron = new Neuron(randomWeights, numInputs, learningRate);
	MaxIteractions = maxIteractions;
	MinQuadraticError = minQuadraticError;
	NumInputs = numInputs;
	delete []randomWeights;
}

NNL::Perceptron::SLP::PerceptronNetwork::~PerceptronNetwork()
{
	delete MainNeuron;
}
