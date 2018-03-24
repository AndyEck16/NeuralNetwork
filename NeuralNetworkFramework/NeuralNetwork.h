#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "NeuralNode.h"
#include "NeuralLayer.h"
#include <vector>
#include <string>
#include <functional>

class NeuralLayer;

class NeuralNetwork
{
public:

	NeuralNetwork();
	~NeuralNetwork();
	
	double REGULARIZATION_WEIGHT;
	double LEARNING_STEP_SIZE;

	void Initialize(std::vector<int> &inNodesPerLayer); //Setup and initialize a NN with the specified nodes per layer

	int NumLayers(); //Return total number of layers in NN, including input and output layers

	void SetHiddenLayerActivationFunctions(
		std::function<double(double)> const &activFunc, 
		std::function<double(double)> const &derivOfActivFunc );

	void SetOutputLayerActivationFunctions(
		std::function<double(double)> const &activFunc, 
		std::function<double(double)> const &derivOfActivFunc );

	void TrainNetwork(std::string trainingDataFilename, int dimensionalityOfInputVector, int dimensionalityOfOutputVector); //Read in a csv file of training vectors and train the neural network off them
	std::vector<double> TestNetwork(std::string trainingDataFilename, int dimensionalityOfInputVector, int dimensionalityOfOutputVector); //Read in a csv file of test vectors and test the NN's predictions against them. Returns fraction of successful predictions for each output node.

	std::vector<double> GetNetworkOutputFromInputVector(std::vector<double> &inputVector);
		

private:
	std::vector<int> nodesPerLayer;
	
	void FeedForwardFromInputVector(std::vector<double> &inputVector);
	std::vector<double> GetCurrentOutputVector();
	void LearnFromSingleInputOutputVectorPair(std::vector<double> &inputVector, std::vector<double> &outputVector);
	std::vector<NeuralLayer*> layers;	
	std::vector<double> GetErrorVectorFromInputOutputPair(std::vector<double> &inputVector, std::vector<double> &outputVector);	

	
	
};

#endif

