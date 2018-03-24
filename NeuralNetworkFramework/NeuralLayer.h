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
	friend NeuralNetwork;	

public:
	
	NeuralLayer();
	NeuralLayer(int numNodesInLayer,
		std::function<double(double)> const& activationFunctionForNodes = NNUtilityFunctions::linearFunc,
		std::function<double(double)> const& derivOfActivationFunctionForNodes = NNUtilityFunctions::linearFunc_deriv);
	~NeuralLayer();	

	int id;

	int NumNodes();
	int NumNodesInPrevLayer();
	std::vector<double> GetNodeValues();
	std::vector<double> GetPrevLayerNodeValues();
	void SetLayerActivationFunctions(
		std::function<double(double)> const &activFunc,
		std::function<double(double)> const &derivOfActivFunc);

private:
	NeuralLayer * prevLayer = NULL;
	NeuralLayer* nextLayer = NULL;
	std::vector<NeuralNode*> nodes;
	NeuralNetwork* parentNetwork;

	void SetPrevLayer(NeuralLayer* newPrevLayer);
	void SetNextLayer(NeuralLayer* newNextLayer);
	void ResetNodeWeightVectors();
	void FeedForward();
	void SetNodeValues(std::vector<double> &newNodeValues);
	std::vector<double> LearnWeightsFromErrorVector(std::vector<double> &errorVector);
};



#endif