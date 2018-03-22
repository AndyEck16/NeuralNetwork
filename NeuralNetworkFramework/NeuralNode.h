#ifndef NEURAL_NODE_H
#define NEURAL_NODE_H

#include "NeuralLayer.h"
#include "NNUtilityFunctions.h"
#include <functional>
#include <vector>

class NeuralLayer;

class NeuralNode {
private:

public:
	NeuralNode();
	NeuralNode(std::function<double(double)> const& activationFunction,
		std::function<double(double)> const& derivOfActivationFunction);
	~NeuralNode();

	std::function<double(double)> activationFunction;
	std::function<double(double)> derivOfActivationFunction;

	NeuralLayer* parentLayer;

	double value;
	int id; //Give node an id

	std::vector<double> weightsFromPreviousLayer;
	double bias;

	void InitializeWeightVector(int sizeOfWeightVector);
	double DotProductOfWeightsAndPreviousLayerNodeValues();

	void UpdateValueFromPrevLayer();
	

	//Error handling functions
	void ThrowNodeWeightSizeMismatchError();
};

#endif