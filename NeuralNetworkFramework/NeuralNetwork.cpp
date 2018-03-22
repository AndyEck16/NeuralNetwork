#include "NeuralNetwork.h"
#include <math.h>
#include "../CSVReader/CSVReader.h"
#include <fstream>
#include <iostream>

NeuralNetwork::NeuralNetwork()
{
	REGULARIZATION_WEIGHT = 0.001;
	LEARNING_STEP_SIZE = 0.001;
}

NeuralNetwork::~NeuralNetwork()
{	
	for (std::vector<NeuralLayer*>::iterator iter = layers.begin(); iter != layers.end(); ++iter) {
		delete *iter;
	}
}


void NeuralNetwork::Initialize(std::vector<int> &inNodesPerLayer) {
	/*
	Initialize an NN with a specified structure of nodes
	*/
	nodesPerLayer = inNodesPerLayer;

	//Initialize the layers of our Neural Network one at a time.
	if (nodesPerLayer.size() > 0) {
		layers = std::vector<NeuralLayer*>(inNodesPerLayer.size());
		//For each layer, initialize it with the proper number of nodes and link it to the adjacent layers in the network.
		layers[0] = new NeuralLayer(inNodesPerLayer[0]);
		layers[0]->id = 0;
		layers[0]->parentNetwork = this;
		for (int i = 1; i < inNodesPerLayer.size(); i++) {
			layers[i] = new NeuralLayer(inNodesPerLayer[i]);
			layers[i]->id = i;
			layers[i]->SetPrevLayer(layers[i - 1]);
			layers[i - 1]->SetNextLayer(layers[i]);
			layers[i]->parentNetwork = this;
		}
	}
}


std::vector<double> NeuralNetwork::TestNetwork(std::string testDataFilename, int dimensionsOfInputVector, int dimensionsOfOutputVector) {
	/*
	Read data vectors from a .csv file and test them against a trained NN. 
	File is located at 'testDataFilename'
	The csv file should be structured so that each line is a training vector.
	The first 'dimensionsOfInputVector' entries on a line should be the elements of a given input vector
	The following 'dimensionsOfOutputVector' entries should be the elements of the output vector produced by the preceding input vector

	csv must only contain commas, data, and newlines.
	currently cannot handle non-numeric data.
	NOTE: The NN must be initialized so that the number of nodes in the first layer is 'dimensionsOfInputVector', and the number of nodes in the last layer is 'dimensionsOfOutputVector'
	*/
	std::ifstream file(testDataFilename);

	std::vector<double> testingVector;
	int numPoints = 0;
	std::vector<double> successfulPredictions(dimensionsOfOutputVector);

	while (CSVReader::ReadLineIntoDoubleVector(file, testingVector)) {
		if (testingVector.size() != dimensionsOfInputVector + dimensionsOfOutputVector) throw std::invalid_argument("ERROR: Data in csv file does not match dimensionality specified by neural net!");
		std::vector<double> testInputVector = NNUtilityFunctions::GetInputVectorFromCombinedVector(testingVector, dimensionsOfInputVector);
		std::vector<double> testOutputVector = NNUtilityFunctions::GetOutputVectorFromCombinedVector(testingVector, dimensionsOfInputVector);

		//TODO: DELETE THIS AFTER GENERALIZING NETWORK FOR ALL KINDS OF OUTPUT VECTORS
		//In this particular NN implementation, we use an activation function at each neuron specialized for BINARY classification.
		//Using this functional requires us to force the outputs to +/- 1
		for (int i = 0; i < testOutputVector.size(); i++) {
			testOutputVector[i] = testOutputVector[i] > 0 ? 1 : -1;
		}

		// Feed forward training point with current weights to get the network's output from this input vector.
		FeedForwardFromInputVector(testInputVector);
		std::vector<double> y_predicted = GetCurrentOutputVector();

		for (int outputNode = 0; outputNode < dimensionsOfOutputVector; outputNode++) {
			//Get sign of each output nodes value to make a prediction on the class.
			y_predicted[outputNode] = y_predicted[outputNode] >= 0 ? 1 : -1;
			successfulPredictions[outputNode] += abs(y_predicted[outputNode] - testOutputVector[outputNode]) < 0.001 ? 1 : 0; // Add a successful prediction if our guess is within 0.001 of actual. (Not strict equality in case of roundoff / precision errors.)
		}
		numPoints++;
	}

	std::vector<double> successFraction(dimensionsOfOutputVector);
	for (int outputNode = 0; outputNode < dimensionsOfOutputVector; outputNode++) {
		successFraction[outputNode] = successfulPredictions[outputNode] / numPoints;
	}

	return successFraction;
}

void NeuralNetwork::TrainNetwork(std::string testDataFilename, int dimensionsOfInputVector, int dimensionsOfOutputVector) {
	/*
	Read data vectors from a .csv file and use them to train an NN.
	File is located at 'testDataFilename'
	The csv file should be structured so that each line is a training vector.
	The first 'dimensionsOfInputVector' entries on a line should be the elements of a given input vector
	The following 'dimensionsOfOutputVector' entries on that line should be the elements of the output vector produced by the preceding input vector

	csv must only contain commas, data, and newlines.
	currently cannot handle non-numeric data.

	NOTE: The NN must be initialized so that the number of nodes in the first layer is 'dimensionsOfInputVector', and the number of nodes in the last layer is 'dimensionsOfOutputVector'
	*/
	std::ifstream file(testDataFilename);

	std::vector<double> trainingVector;

	int numPoints = 0;
	while (CSVReader::ReadLineIntoDoubleVector(file, trainingVector)) {

		if (trainingVector.size() != dimensionsOfInputVector + dimensionsOfOutputVector) throw std::invalid_argument("ERROR: Data in csv file does not match dimensionality specified by neural net!");
		std::vector<double> trainInputVector = NNUtilityFunctions::GetInputVectorFromCombinedVector(trainingVector, dimensionsOfInputVector);
		std::vector<double> trainOutputVector = NNUtilityFunctions::GetOutputVectorFromCombinedVector(trainingVector, dimensionsOfInputVector);

		//TODO: DELETE THIS AFTER GENERALIZING NETWORK FOR ALL KINDS OF OUTPUT VECTORS
		//In this particular NN implementation, we use an activation function at each neuron specialized for BINARY classification.
		//Using this functional requires us to force the outputs to +/- 1
		for (int i = 0; i < trainOutputVector.size(); i++) {
			trainOutputVector[i] = trainOutputVector[i] > 0 ? 1 : -1;
		}

		LearnFromSingleInputOutputVectorPair(trainInputVector, trainOutputVector);

		numPoints++;

		if (numPoints % 10000 == 0) {
			std::cout << numPoints << " vectors trained on." << std::endl;
		}

	}
}



void NeuralNetwork::LearnFromSingleInputOutputVectorPair(std::vector<double> &inputVector, std::vector<double> &outputVector) {
	/*
	Train the weights of an NN using a single input and resulting output vector	
	NOTE: The NN must be initialized so that the number of nodes in the first layer is 'dimensionsOfInputVector', and the number of nodes in the last layer is 'dimensionsOfOutputVector'
	*/
	if (inputVector.size() != nodesPerLayer[0]) throw std::invalid_argument("ERROR: input vector size and # of nodes in NN 1st layer must match");
	if (outputVector.size() != nodesPerLayer[NumLayers() - 1]) throw std::invalid_argument("ERROR: output vector size and # of nodes in NN last layer must match");

	std::vector<double> errorVector = GetErrorVectorFromInputOutputPair(inputVector, outputVector);

	NeuralLayer* currentLayer = *(layers.end()-1);
	while (currentLayer) {
		errorVector = currentLayer->LearnWeightsFromErrorVector(errorVector);
		currentLayer = currentLayer->GetPrevLayer();
	}

}

std::vector<double> NeuralNetwork::GetErrorVectorFromInputOutputPair(std::vector<double> &trainingInputVector, std::vector<double> &trainingOutputVector) {
	//Feed input vector into network. Compare output to expected values from training output vector
	FeedForwardFromInputVector(trainingInputVector);
	std::vector<double> y_predicted = GetCurrentOutputVector();
	std::vector<double> errorVector(trainingOutputVector.size());

	for (int outputNode = 0; outputNode < trainingOutputVector.size(); outputNode++) {
		errorVector[outputNode] =trainingOutputVector[outputNode] - y_predicted[outputNode];
	}

	return errorVector;
}

/*
std::vector<std::vector<std::vector<double>>> NeuralNetwork::getWeightsErrorGradient(std::vector<double> &inputVector, std::vector<double> &outputVector) {
	
	if (inputVector.size() != nodesPerLayer[0]) throw std::invalid_argument("ERROR: input vector size and # of nodes in NN 1st layer must match");
	if (outputVector.size() != nodesPerLayer[NumLayers() - 1]) throw std::invalid_argument("ERROR: output vector size and # of nodes in NN last layer must match");

	std::vector<std::vector<std::vector<double>>> gradient_sum(NumLayers()-1);
	std::vector<std::vector<double>> ess;
	std::vector<double> y_predicted;

	//In this particular NN implementation, we use an activation function at each neuron specialized for BINARY classification.
	//Using this functional requires us to force the outputs to +/- 1
	for (int i = 0; i < outputVector.size(); i++) {
		outputVector[i] = outputVector[i] > 0 ? 1 : -1;
	}

	//Initialize gradient_sum size
	for (int i = 0; i < NumLayers() - 1; i++) {	
		int layer = i + 1;
		gradient_sum[i] = std::vector<std::vector<double>>(nodesPerLayer[layer]);
		for (int nodeInLayer = 0; nodeInLayer < nodesPerLayer[layer]; nodeInLayer++) {
			gradient_sum[i][nodeInLayer] = std::vector<double>(nodesPerLayer[layer-1] + 1); //1 slot for each node in previous layer, plus 1 slot for a constant
		}
	}

	// Feed forward training point with current weights to get the network's output from this input vector.
	FeedForwardFromInputVector(inputVector);
	y_predicted = GetCurrentOutputVector();

	//Use this output and backpropogate to determine the updates we should
	//apply to the weights.Start first from the output layer.

	//Find gradient vector at output nodes
	std::vector<double> delta(y_predicted.size());
	NeuralLayer * const currentLayer = *(layers.end());
	NeuralLayer * const previousLayer = *(layers.end() - 1);
	for (int i = 0; i < y_predicted.size(); i++) {
		NeuralNode * const currentNode = currentLayer->nodes[i];
		std::vector<double> prevLayerNodeValues = previousLayer->GetNodeValues();

		double innerProductAtNode = NNUtilityFunctions::innerProduct(prevLayerNodeValues, currentNode->weightsFromPreviousLayer);
		delta[i] = -2 * (outputVector[i] - y_predicted[i])
			* currentNode->derivOfActivationFunction(innerProductAtNode);

		//Add this gradient to our total for the data set and add regularization cost.
		int prevVectorSize = previousLayer->NumNodes();
		for (int j = 0; j < prevVectorSize; j++) {
			gradient_sum[NumLayers()-2][i][j] = gradient_sum[NumLayers() - 2][i][j] + delta[i] * prevLayerNodeValues[j]
				+ REGULARIZATION_WEIGHT * currentNode->weightsFromPreviousLayer[j];
		}
	}

	std::vector<double> weightSumTerm;
	//Find gradient vector for each node in hidden layers by backpropogation.
	ess = std::vector<std::vector<double>>(NumLayers() - 1);
	for (int hiddenLayer = nodesPerLayer.size() - 2; hiddenLayer > 0; hiddenLayer--) {
		//Compute intermediate result, 'weightSumTerm' to multiply the functional gradient by
		weightSumTerm = std::vector<double>(nodesPerLayer[hiddenLayer]);
		for (int nextLayerNode = 0; nextLayerNode < nodesPerLayer[hiddenLayer + 1]; nextLayerNode++) {
			for (int currentLayerNode = 0; currentLayerNode < nodesPerLayer[hiddenLayer]; currentLayerNode++) {
				if (hiddenLayer == nodesPerLayer.size() - 2) {

					weightSumTerm[currentLayerNode] += delta[nextLayerNode] * nodeWeights[nodeWeights.size() - 1][nextLayerNode][currentLayerNode];
				}
				else {
					weightSumTerm[currentLayerNode] += ess[hiddenLayer + 1][nextLayerNode] * nodeWeights[hiddenLayer + 1][nextLayerNode][currentLayerNode];
				}
			}
		}

		ess[hiddenLayer] = std::vector<double>(nodesPerLayer[hiddenLayer]);
		for (int hiddenNode = 0; hiddenNode < nodesPerLayer[hiddenLayer]; hiddenNode++) {

			ess[hiddenLayer][hiddenNode] = sigmoidFunc_deriv(networkState[hiddenLayer - 1], nodeWeights[hiddenLayer - 1][hiddenNode]) * weightSumTerm[hiddenNode];
			for (int prevNode = 0; prevNode < nodesPerLayer[hiddenLayer - 1]; prevNode++) {
				gradient_sum[hiddenLayer - 1][hiddenNode][prevNode] = gradient_sum[hiddenLayer - 1][hiddenNode][prevNode]
					+ ess[hiddenLayer][hiddenNode] * networkState[hiddenLayer - 1][prevNode]
					+ REGULARIZATION_WEIGHT * nodeWeights[hiddenLayer - 1][hiddenNode][prevNode];
			}

		}
	}

	return gradient_sum;
}
*/

int NeuralNetwork::NumLayers() {
	//Utility function to return the number of layers in the NN, including input and output layers.
	return nodesPerLayer.size();
}

void NeuralNetwork::FeedForwardFromInputVector(std::vector<double> &inputVector)
{	
	if (inputVector.size() != nodesPerLayer[0]) throw std::invalid_argument("ERROR: input vector size and NN 1st layer size must match");
	std::vector<std::vector<double>> newNodeValues(NumLayers());
	newNodeValues[0] = inputVector;
	
	layers[0]->SetNodeValues(inputVector);
	NeuralLayer* currentLayer = layers[0];
	while (currentLayer->GetNextLayer()) {
		currentLayer->FeedForward();
		currentLayer = currentLayer->GetNextLayer();
	}
}

std::vector<double> NeuralNetwork::GetCurrentOutputVector() {
	NeuralLayer* lastLayer = *(layers.end()-1);
	return lastLayer->GetNodeValues();
}



