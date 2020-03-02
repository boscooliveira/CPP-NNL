#include "Neuron.h"

using namespace NNL::Perceptron::SLP;

__inline float CalculateError(float desiredResult, float obtainedResult)
{
	return desiredResult - obtainedResult;
}

__inline float CalculateNewWeight(float curentWeight, float input, float error, float learningRate)
{
	return curentWeight + learningRate * error * input;
}

float NNL::Perceptron::SLP::Neuron::ProcessInputs(float* inputs) const
{
	float sum = Bias;
	for (int i = 0; i < NumInputs; i++)
	{
		sum += Weights[i] * inputs[i];
	}
	return sum > 0 ? 1 : 0;
}

float NNL::Perceptron::SLP::Neuron::Train(float* inputs, float desiredResult)
{
	float obtainedResult = ProcessInputs(inputs);
	float error = CalculateError(desiredResult, obtainedResult);

	float *endPosition = Weights + NumInputs;
	for (float *weigth = Weights; weigth < endPosition; weigth++, inputs++)
	{
		*weigth = CalculateNewWeight(*weigth, *inputs, error, LearningRate);
	}

	Bias = CalculateNewWeight(Bias, 1, error, LearningRate);
	return error * error;
}

NNL::Perceptron::SLP::Neuron::~Neuron()
{
	delete Weights;
}

NNL::Perceptron::SLP::Neuron::Neuron(float* initialWeights, int numInputs, float learningRate)
{
	NumInputs = numInputs;
	Weights = new float[NumInputs];
	for (int i = 0; i < numInputs; i++)
	{
		Weights[i] = initialWeights[i];
	}
	LearningRate = learningRate;
	Bias = 0;
}
