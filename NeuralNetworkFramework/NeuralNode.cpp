#include "NeuralNode.h"
#include <string>

NeuralNode::NeuralNode() {
	//Default to linear activation function if none is specified.
	activationFunction = NNUtilityFunctions::linearFunc;
	derivOfActivationFunction = NNUtilityFunctions::linearFunc_deriv;
}

NeuralNode::NeuralNode(std::function<double(double)> const& inActivFunc, std::function<double(double)> const& inDerivOfActivFunc) {
	activationFunction = inActivFunc;
	derivOfActivationFunction = inDerivOfActivFunc;
}

NeuralNode::~NeuralNode() {
}

void NeuralNode::InitializeWeightVector(int sizeOfWeightVector) {
	weightsFromPreviousLayer = std::vector<double>(sizeOfWeightVector);
}

void NeuralNode::ThrowNodeWeightSizeMismatchError() {
	std::string errorString = "ERROR: Weight vector and # of nodes are different sizes! \nCurrent layer id: " + std::to_string(parentLayer->id)
		+ "\nNode id: " + std::to_string(id)
		+ "\nSize of weight vector at node: " + std::to_string(weightsFromPreviousLayer.size())
		+ "\nNumber of nodes in previous layer: " + std::to_string(parentLayer->NumNodesInPrevLayer());
	throw std::exception(errorString.c_str());
}

double NeuralNode::DotProductOfWeightsAndPreviousLayerNodeValues() {
	double dotProd;

	//Check that the vector of weights for this node matches the number of nodes in the current layer.
	if (weightsFromPreviousLayer.size() != parentLayer->NumNodesInPrevLayer()) {
		ThrowNodeWeightSizeMismatchError();
	}

	//Compute dot product of current node values and weights in this layer	
	dotProd = NNUtilityFunctions::innerProduct(parentLayer->GetPrevLayerNodeValues(), weightsFromPreviousLayer);
	dotProd += bias; //Add node's bias to dot product before inputting into activation function.

	return dotProd;
}

void NeuralNode::UpdateValueFromPrevLayer() {
	double dotProd = DotProductOfWeightsAndPreviousLayerNodeValues();
	value = activationFunction(dotProd);
}