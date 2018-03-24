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
	
	nodesPerLayer = inNodesPerLayer;

	if (nodesPerLayer.size() > 0) {
		layers = std::vector<NeuralLayer*>(inNodesPerLayer.size());
		layers[0] = new NeuralLayer(inNodesPerLayer[0]);
		layers[0]->id = 0;
		layers[0]->parentNetwork = this;
		for (int i = 1; i < (int)(inNodesPerLayer.size()); i++) {
			layers[i] = new NeuralLayer(inNodesPerLayer[i]);
			layers[i]->id = i;
			layers[i]->SetPrevLayer(layers[i - 1]);
			layers[i - 1]->SetNextLayer(layers[i]);
			layers[i]->parentNetwork = this;
		}
	}
}

void NeuralNetwork::SetHiddenLayerActivationFunctions(
	std::function<double(double)> const &activFunc, 
	std::function<double(double)> const &derivOfActivFunc) 
{
	if (layers.size() > 2) {
		int layerIdx = 1;
		int lastLayerIdx = layers.size() - 1;
		while (layerIdx != lastLayerIdx) {
			layers[layerIdx]->SetLayerActivationFunctions(activFunc, derivOfActivFunc);
			layerIdx++;
		}
	}
}

void NeuralNetwork::SetOutputLayerActivationFunctions(
	std::function<double(double)> const &activFunc, 
	std::function<double(double)> const& derivOfActivFunc) 
{
	if (layers.size() > 1) {
		int lastLayerIdx = layers.size() - 1;
		layers[lastLayerIdx]->SetLayerActivationFunctions(activFunc, derivOfActivFunc);
	}
}

std::vector<double> NeuralNetwork::GetNetworkOutputFromInputVector(std::vector<double> &inputVector) {
	FeedForwardFromInputVector(inputVector);
	return GetCurrentOutputVector();
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

		// Feed forward training point with current weights to get the network's output from this input vector.
		std::vector<double> y_predicted = GetNetworkOutputFromInputVector(testInputVector);

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
		currentLayer = currentLayer->prevLayer;
	}

}

std::vector<double> NeuralNetwork::GetErrorVectorFromInputOutputPair(std::vector<double> &trainingInputVector, std::vector<double> &trainingOutputVector) {
	//Feed input vector into network. Compare output to expected values from training output vector
	FeedForwardFromInputVector(trainingInputVector);
	std::vector<double> y_predicted = GetCurrentOutputVector();
	std::vector<double> errorVector(trainingOutputVector.size());

	for (int outputNode = 0; outputNode < (int)(trainingOutputVector.size()); outputNode++) {
		errorVector[outputNode] =trainingOutputVector[outputNode] - y_predicted[outputNode];
	}

	return errorVector;
}

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
	while (currentLayer->nextLayer) {
		currentLayer->FeedForward();
		currentLayer = currentLayer->nextLayer;
	}
}

std::vector<double> NeuralNetwork::GetCurrentOutputVector() {
	NeuralLayer* lastLayer = *(layers.end()-1);
	return lastLayer->GetNodeValues();
}



