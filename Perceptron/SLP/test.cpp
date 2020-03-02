#include "PerceptronNetwork.h"
#include <cstdlib>
#include <cstdio>

using namespace NNL::Perceptron::SLP;

int main()
{
	float trainingSet[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
	float results[4] = { 0, 0, 0, 1 };
	PerceptronNetwork network(2,0.05,100,0);

	printf("\nr1 %5f \n", network.GetResult(((float*)trainingSet)));
	printf("r2 %5f \n", network.GetResult(((float*)trainingSet) + 2));
	printf("r3 %5f \n", network.GetResult(((float*)trainingSet) + 4));
	printf("r4 %5f \n", network.GetResult(((float*)trainingSet) + 6));

	network.Train(4, (float *)trainingSet, results);

	printf("\nr1 %5f \n", network.GetResult(((float*)trainingSet)));
	printf("r2 %5f \n", network.GetResult(((float*)trainingSet) + 2));
	printf("r3 %5f \n", network.GetResult(((float*)trainingSet) + 4));
	printf("r4 %5f \n", network.GetResult(((float*)trainingSet) + 6));
	return 0;
}