#include "Neuron.h"
#pragma once

namespace NNL 
{
	namespace Perceptron 
	{
		namespace SLP 
		{
			class PerceptronNetwork 
			{
			private:
				Neuron* MainNeuron;
				int MaxIteractions;
				float MinQuadraticError;
				int NumInputs;

			public:
				float GetResult(float* inputs) const;
				void Train(int trainSetQtt, float* trainInputs, float *trainResults);
				PerceptronNetwork(int numInputs, float learningRate, int maxIteractions, float minQuadraticError);
				~PerceptronNetwork();

			};
		}
	}
}
