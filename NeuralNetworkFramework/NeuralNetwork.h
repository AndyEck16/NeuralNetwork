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

	//Constructor / Destructors
	NeuralNetwork();
	~NeuralNetwork();

	//Attributes
	std::vector<int> nodesPerLayer;
	double REGULARIZATION_WEIGHT;
	double LEARNING_STEP_SIZE;

	//Methods
	void Initialize(std::vector<int> &inNodesPerLayer); //Setup and initialize a NN with the specified nodes per layer
	void TrainNetwork(std::string trainingDataFilename, int dimensionalityOfInputVector, int dimensionalityOfOutputVector); //Read in a csv file of training vectors and train the neural network off them
	std::vector<double> TestNetwork(std::string trainingDataFilename, int dimensionalityOfInputVector, int dimensionalityOfOutputVector); //Read in a csv file of test vectors and test the NN's predictions against them. Returns fraction of successful predictions for each output node.
	int NumLayers(); //Return total number of layers in NN, including input and output layers
	void FeedForwardFromInputVector(std::vector<double> &inputVector);
	std::vector<double> GetCurrentOutputVector();
	void LearnFromSingleInputOutputVectorPair(std::vector<double> &inputVector, std::vector<double> &outputVector);
	

private:

	std::vector<NeuralLayer*> layers;
	
	std::vector<double> GetErrorVectorFromInputOutputPair(std::vector<double> &inputVector, std::vector<double> &outputVector);
	
	std::vector<std::vector<std::vector<double>>> getWeightsErrorGradient(std::vector<double> &inputVector, std::vector<double> &outputVector);

	
	
};

#endif

