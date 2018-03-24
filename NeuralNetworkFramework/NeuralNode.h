#ifndef NEURAL_NODE_H
#define NEURAL_NODE_H

#include "NeuralLayer.h"
#include "NNUtilityFunctions.h"
#include <functional>
#include <vector>

class NeuralLayer;

class NeuralNode {
	friend NeuralLayer;

public:
	NeuralNode();
	NeuralNode(std::function<double(double)> const& activationFunction,
		std::function<double(double)> const& derivOfActivationFunction);
	~NeuralNode();	
	int id; //Give node an id	

	double value;

	void InitializeWeightVector(int sizeOfWeightVector);
	void UpdateValueFromPrevLayer();	

private:
	std::function<double(double)> activationFunction;
	std::function<double(double)> derivOfActivationFunction;

	NeuralLayer* parentLayer;

	std::vector<double> weightsFromPreviousLayer;
	double bias;

	double DotProductOfWeightsAndPreviousLayerNodeValues();
	void ThrowNodeWeightSizeMismatchError();
	
};

#endif