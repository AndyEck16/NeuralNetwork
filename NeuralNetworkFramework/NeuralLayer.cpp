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

int NeuralLayer::NumNodes() const{
	return nodes.size();
}

int NeuralLayer::NumNodesInPrevLayer() {
	if (prevLayer)
		return prevLayer->NumNodes();
	else
		return 0;
}

void NeuralLayer::SetNextLayer(NeuralLayer* newNextLayer) {
	if (newNextLayer != nextLayer) {
		nextLayer = newNextLayer;
	}
}

void NeuralLayer::SetPrevLayer(NeuralLayer* newPrevLayer) {
	if (newPrevLayer != prevLayer) {
		prevLayer = newPrevLayer;
		ResetNodeWeightVectors();
	}
}

void NeuralLayer::ResetNodeWeightVectors() {
	if (prevLayer) {
		for (int nodeIdx = 0; nodeIdx < NumNodes(); nodeIdx++) {
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
	for (int i = 0; i < NumNodes(); i++) {
		nodeValueVector[i] = nodes[i]->value;
	}
	return nodeValueVector;
}

std::vector<double> NeuralLayer::GetPrevLayerNodeValues() {
	std::vector<double> nodeValueVector;
	if (prevLayer) {
		nodeValueVector = std::vector<double>(NumNodesInPrevLayer());
		for (int i = 0; i < NumNodesInPrevLayer(); i++) {
			nodeValueVector[i] = prevLayer->nodes[i]->value;
		}
	}
	return nodeValueVector;
}

NeuralLayer const* const NeuralLayer::GetPrevLayer() {
	NeuralLayer const * const returnPrevLayer = prevLayer;
	return returnPrevLayer;
}

void NeuralLayer::SetNodeValues(std::vector<double> &newNodeValues) {
	if (newNodeValues.size() != NumNodes()) throw std::exception("ERROR: Size of new node values vector must match current size of neural layer");
	
	for (int nodeIdx = 0; nodeIdx < NumNodes(); nodeIdx++) {
		nodes[nodeIdx]->value = newNodeValues[nodeIdx];
	}
}

void NeuralLayer::SetLayerActivationFunctions(
		std::function<double(double)> const &activFunc,
		std::function<double(double)> const &derivOfActivFunc) 
{
	for (NeuralNode* theNode : nodes) {
		theNode->activationFunction = activFunc;
		theNode->derivOfActivationFunction = derivOfActivFunc;
	}
}

std::vector<double> NeuralLayer::LearnWeightsFromErrorVector(std::vector<double> &errorVector) {
	if (errorVector.size() != NumNodes()) throw std::exception("ERROR: errorVector is not same size as # of nodes in layer");
	double learningStepSize = parentNetwork->LEARNING_STEP_SIZE;
	double regularizationWeight = parentNetwork->REGULARIZATION_WEIGHT;
	
	std::vector<double> backPropogatedErrorVector;

	if (prevLayer) {
		std::vector<double> prevLayerNodeValues = prevLayer->GetNodeValues();
		backPropogatedErrorVector = std::vector<double>(prevLayer->NumNodes());

		for (int i = 0; i < NumNodes(); i++) {
			NeuralNode * const currentNode = nodes[i];

			double innerProductAtNode = currentNode->DotProductOfWeightsAndPreviousLayerNodeValues();
			double currentNodeErrorGradientTerm;
			currentNodeErrorGradientTerm = 2 * errorVector[i] * currentNode->derivOfActivationFunction(innerProductAtNode);		

			//Alter each value in the weight vector at the current node using the error term and the regularization cost
			int prevLayerSize = prevLayer->NumNodes();
			for (int prevLayerNode = 0; prevLayerNode < prevLayerSize; prevLayerNode++) {
				backPropogatedErrorVector[prevLayerNode] += (1.0 / 2.0) * currentNodeErrorGradientTerm * currentNode->weightsFromPreviousLayer[prevLayerNode];

				double gradientTerm = currentNodeErrorGradientTerm * prevLayerNodeValues[prevLayerNode]
										+ regularizationWeight * currentNode->weightsFromPreviousLayer[prevLayerNode];
				currentNode->weightsFromPreviousLayer[prevLayerNode] = currentNode->weightsFromPreviousLayer[prevLayerNode] 
																		- learningStepSize * gradientTerm;				
			}
			//Alter bias using error term and regularization cost
			double gradientTerm = currentNodeErrorGradientTerm + regularizationWeight * currentNode->bias;
			currentNode->bias = currentNode->bias - learningStepSize * gradientTerm;
		}
	}
	return backPropogatedErrorVector;
}

double NeuralLayer::DotProductWithWeightVector(std::vector<double> const &weightVector) const {
	if (weightVector.size() != NumNodes()) throw std::exception("ERROR: weightVector is not same size as # of nodes in layer");
	double dotProd = 0;
	for (int i = 0; i < NumNodes(); i++) {
		dotProd += weightVector[i] * nodes[i]->value;
	}
	return dotProd;
}