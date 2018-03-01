#pragma once

#include <vector>
#include <string>


class NeuralNetwork
{
public:
	//Constructor / Destructors
	NeuralNetwork();
	~NeuralNetwork();

	//Attributes
	std::vector<int> nodesPerLayer;

	//Methods
	void Initialize(std::vector<int> &inNodesPerLayer); //Setup and initialize a NN with the specified nodes per layer
	void TrainNetwork(std::string trainingDataFilename, int dimensionalityOfInputVector, int dimensionalityOfOutputVector); //Read in a csv file of training vectors and train the neural network off them
	std::vector<double> TestNetwork(std::string trainingDataFilename, int dimensionalityOfInputVector, int dimensionalityOfOutputVector); //Read in a csv file of test vectors and test the NN's predictions against them. Returns fraction of successful predictions for each output node.
	int NumLayers(); //Return total number of layers in NN, including input and output layers
	std::vector<std::vector<double>> FeedForwardFromInputVector(std::vector<double> &inputVector);
	void LearnFromSingleInputOutputVectorPair(std::vector<double> &inputVector, std::vector<double> &outputVector);
	

private:
	double REGULARIZATION_WEIGHT;
	double LEARNING_STEP_SIZE;	
	std::vector<std::vector<std::vector<double>>> getWeightsErrorGradient(std::vector<double> &inputVector, std::vector<double> &outputVector);
	std::vector<std::vector<std::vector<double>>> nodeWeights;
	double sigmoidFunc(std::vector<double> &x, std::vector<double> &weights);
	double sigmoidFunc_deriv(std::vector<double> &x, std::vector<double> &weights);
	double gFun(std::vector<double> &x, std::vector<double> &weights);
	double gFun_prime(std::vector<double> &x, std::vector<double> &weights);
	
};

