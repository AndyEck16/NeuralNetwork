#include "NeuralLayer.h"
#include <string>


NeuralLayer::NeuralLayer() {

}

NeuralLayer::NeuralLayer(int numNodesInLayer, std::function<double(double)> const& activationFunc, std::function<double(double)> const& derivOfActivationFunc) {
	nodes = std::vector<NeuralNode*>(numNodesInLayer);
	for (int i = 0; i < numNodesInLayer; i++) {
		nodes[i] = new NeuralNode(activationFunc, derivOfActivationFunc);
		nodes[i]->parentLayer = this;
	}
}


NeuralLayer::~NeuralLayer() {
	for (std::vector<NeuralNode*>::iterator nodeIter = nodes.begin(); nodeIter != nodes.end(); ++nodeIter) {
		delete *nodeIter;
	}
}

NeuralLayer* NeuralLayer::GetNextLayer() {
	return nextLayer;
}

NeuralLayer* NeuralLayer::GetPrevLayer() {
	return prevLayer;
}

void NeuralLayer::SetNextLayer(NeuralLayer* newNextLayer) {
	if (newNextLayer != nextLayer) {
		nextLayer = newNextLayer;
	}
}

int NeuralLayer::NumNodes() {
	return nodes.size();
}

void NeuralLayer::SetPrevLayer(NeuralLayer* newPrevLayer) {
	if (newPrevLayer != prevLayer) {
		prevLayer = newPrevLayer;
		ResetNodeWeightVectors();
	}
}

void NeuralLayer::ResetNodeWeightVectors() {
	if (prevLayer) {
		for (int nodeIdx = 0; nodeIdx < nodes.size(); nodeIdx++) {
			nodes[nodeIdx]->InitializeWeightVector(prevLayer->NumNodes());
		}
	}
}

void NeuralLayer::FeedForward() {
	if (nextLayer) {
		NeuralNode* nextLayerNode;
		for (std::vector<NeuralNode*>::iterator nextLayerNodeIter = nextLayer->nodes.begin(); nextLayerNodeIter != nextLayer->nodes.end(); ++nextLayerNodeIter) {
			nextLayerNode = *nextLayerNodeIter;			
			nextLayerNode->UpdateValueFromPrevLayer();
		}
	}
}


std::vector<double> NeuralLayer::GetNodeValues() {
	std::vector<double> nodeValueVector(nodes.size());
	for (int i = 0; i < nodes.size(); i++) {
		nodeValueVector[i] = nodes[i]->value;
	}

	return nodeValueVector;
}

void NeuralLayer::SetNodeValues(std::vector<double> &newNodeValues) {
	for (int nodeIdx = 0; nodeIdx < newNodeValues.size(); nodeIdx++) {
		nodes[nodeIdx]->value = newNodeValues[nodeIdx];
	}
}

std::vector<double> NeuralLayer::LearnWeightsFromErrorVector(std::vector<double> &errorVector) {
	double learningStepSize = parentNetwork->LEARNING_STEP_SIZE;
	double regularizationWeight = parentNetwork->REGULARIZATION_WEIGHT;
	
	std::vector<double> backPropogatedErrorVector;

	if (prevLayer) {
		std::vector<double> prevLayerNodeValues = prevLayer->GetNodeValues();
		backPropogatedErrorVector = std::vector<double>(prevLayer->NumNodes());

		for (int i = 0; i < errorVector.size(); i++) {
			NeuralNode * const currentNode = nodes[i];

			double innerProductAtNode = currentNode->DotProductOfWeightsAndPreviousLayerNodeValues();
			double currentNodeErrorTerm = -2 * errorVector[i] * currentNode->derivOfActivationFunction(innerProductAtNode);			

			//Alter each value in the weight vector at the current node using the error term and the regularization cost
			int prevLayerSize = prevLayer->NumNodes();
			for (int prevLayerNode = 0; prevLayerNode < prevLayerSize; prevLayerNode++) {
				backPropogatedErrorVector[prevLayerNode] += currentNodeErrorTerm * currentNode->weightsFromPreviousLayer[prevLayerNode];

				double gradientTerm = currentNodeErrorTerm * prevLayerNodeValues[prevLayerNode] 
										+ regularizationWeight * currentNode->weightsFromPreviousLayer[prevLayerNode];
				currentNode->weightsFromPreviousLayer[prevLayerNode] = currentNode->weightsFromPreviousLayer[prevLayerNode] 
																		- learningStepSize * gradientTerm;				
			}
			//Alter bias using error term and regularization cost
			double gradientTerm = currentNodeErrorTerm + regularizationWeight * currentNode->bias;
			currentNode->bias = currentNode->bias - learningStepSize * gradientTerm;
		}
	}
	return backPropogatedErrorVector;


}