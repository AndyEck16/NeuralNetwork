#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H

#include "NeuralNode.h"
#include "NeuralNetwork.h"
#include "NNUtilityFunctions.h"
#include <functional>
#include <vector>

class NeuralNetwork;
class NeuralNode;

class NeuralLayer {
	NeuralLayer* prevLayer = NULL;
	NeuralLayer* nextLayer = NULL;
	void ResetNodeWeightVectors();

public:
	//Constructors / Destructor
	NeuralLayer();
	NeuralLayer(int numNodesInLayer,
		std::function<double(double)> const& activationFunctionForNodes = NNUtilityFunctions::linearFunc,
		std::function<double(double)> const& derivOfActivationFunctionForNodes = NNUtilityFunctions::linearFunc_deriv);
	~NeuralLayer();

	int id; //Give layer an id
	NeuralNetwork* parentNetwork;

	//Get / Set neighboring layers
	void SetPrevLayer(NeuralLayer* newPrevLayer);
	void SetNextLayer(NeuralLayer* newNextLayer);
	NeuralLayer* GetPrevLayer();
	NeuralLayer* GetNextLayer();
	//void ResetNodeWeights();

	//Get nodes in current layer
	std::vector<NeuralNode*> nodes;
	int NumNodes();
	std::vector<double> GetNodeValues();
	void FeedForward();
	void SetNodeValues(std::vector<double> &newNodeValues);

	std::vector<double> LearnWeightsFromErrorVector(std::vector<double> &errorVector);

	
};

#endif