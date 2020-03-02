#pragma once

namespace NNL 
{
	namespace Perceptron 
	{
		namespace SLP 
		{
			class Neuron
			{
			private:
				float* Weights;
				float Bias;
				int NumInputs;
				float LearningRate;

			public:
				float ProcessInputs(float* inputs) const;
				Neuron(float* initialWeights, int numInputs, float learningRate);
				float Train(float* inputs, float result);
				~Neuron();
			};
		} // NNL
	} // Preceprton
} // SLP
