#include "NeuralNetwork.h"
#include <math.h>
#include "../CSVReader/CSVReader.h"
#include <fstream>
#include <iostream>

NeuralNetwork::NeuralNetwork()
{
	REGULARIZATION_WEIGHT = 0.001;
	LEARNING_STEP_SIZE = 0.01;
}

NeuralNetwork::~NeuralNetwork()
{
}


void NeuralNetwork::Initialize(std::vector<int> &inNodesPerLayer) {
	/*
	Initialize an NN with a specified structure of nodes
	*/
	nodesPerLayer = inNodesPerLayer;

	//Initialize our 'nodeWeights' matrix.
	//Will be a 3 dim matrix that holds the weight vectors used to calculate the value of one node from the previous layer of nodes.
	//Dim 1: beginning with the first layer after the input layer, how many layers into the network are we? Size is NumLayers()-1
	//Dim 2: In this layer, what node is being calculated? Size is # of nodes in the layer
	//Dim 3: What weight is given to each node in the previous layer before summing them all together and calulating value of the next layer's node? Size is # of nodes in previous layer + 1
	//		 The extra weight slot is added to support adding a constant offset to the vector.
	nodeWeights = std::vector<std::vector<std::vector<double>>>(NumLayers() - 1);
	for (int layer = 1; layer < NumLayers(); layer++) {
		nodeWeights[layer - 1] = std::vector<std::vector<double>>(nodesPerLayer[layer]);
		for (int node = 0; node < nodesPerLayer[layer]; node++) {
			nodeWeights[layer - 1][node] = std::vector<double>(nodesPerLayer[layer - 1] + 1); //We add a 1 because each weight vector is augmented to effectively add a 'constant' node to the previous layer
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
		std::vector<double>::const_iterator inputVectorStart = testingVector.begin();
		std::vector<double>::const_iterator inputVectorEnd = testingVector.begin() + dimensionsOfInputVector;
		std::vector<double> inputVector(inputVectorStart, inputVectorEnd);

		std::vector<double>::const_iterator outputVectorStart = testingVector.begin() + dimensionsOfInputVector;
		std::vector<double>::const_iterator outputVectorEnd = testingVector.end();
		std::vector<double> outputVector(outputVectorStart, outputVectorEnd);

		//In this particular NN implementation, we use an activation function at each neuron specialized for BINARY classification.
		//Using this functional requires us to force the outputs to +/- 1
		for (int i = 0; i < outputVector.size(); i++) {
			outputVector[i] = outputVector[i] > 0 ? 1 : -1;
		}

		// Feed forward training point with current weights to get the network's output from this input vector.
		std::vector<std::vector<double>> networkState = FeedForwardFromInputVector(inputVector);
		std::vector<double> y_predicted = networkState[networkState.size() - 1];


		for (int outputNode = 0; outputNode < dimensionsOfOutputVector; outputNode++) {
			//Get sign of each output nodes value to make a prediction on the class.
			y_predicted[outputNode] = y_predicted[outputNode] >= 0 ? 1 : -1;
			successfulPredictions[outputNode] += abs(y_predicted[outputNode] - outputVector[outputNode]) < 0.001 ? 1 : 0; // Add a successful prediction if our guess is within 0.001 of actual. (Not strict equality in case of roundoff / precision errors.)
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
	The following 'dimensionsOfOutputVector' entries should be the elements of the output vector produced by the preceding input vector

	csv must only contain commas, data, and newlines.
	currently cannot handle non-numeric data.

	NOTE: The NN must be initialized so that the number of nodes in the first layer is 'dimensionsOfInputVector', and the number of nodes in the last layer is 'dimensionsOfOutputVector'
	*/
	std::ifstream file(testDataFilename);

	std::vector<double> trainingVector;

	int numPoints = 0;
	while (CSVReader::ReadLineIntoDoubleVector(file, trainingVector)) {

		if (trainingVector.size() != dimensionsOfInputVector + dimensionsOfOutputVector) throw std::invalid_argument("ERROR: Data in csv file does not match dimensionality specified by neural net!");
		std::vector<double>::const_iterator inputVectorStart = trainingVector.begin();
		std::vector<double>::const_iterator inputVectorEnd = trainingVector.begin() + dimensionsOfInputVector;
		std::vector<double> inputVector(inputVectorStart, inputVectorEnd);

		std::vector<double>::const_iterator outputVectorStart = trainingVector.begin() + dimensionsOfInputVector;
		std::vector<double>::const_iterator outputVectorEnd = trainingVector.end();
		std::vector<double> outputVector(outputVectorStart, outputVectorEnd);

		LearnFromSingleInputOutputVectorPair(inputVector, outputVector);

		numPoints++;

		if (numPoints % 10000 == 0) {
			int x = 0;
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
	
	std::vector<std::vector<std::vector<double>>> outGradUpdate = getWeightsErrorGradient(inputVector, outputVector);

	//Update each weight using the error gradient from this training vector.
	//Multiply the gradient by our 'step_size' and update the weights.
	for (int layer = 0; layer < NumLayers() - 1; layer++) {
		for (int i = 0; i < nodesPerLayer[layer + 1]; i++) {
			for (int j = 0; j < nodesPerLayer[layer] + 1; j++) {
				nodeWeights[layer][i][j] = nodeWeights[layer][i][j] - LEARNING_STEP_SIZE * outGradUpdate[layer][i][j];
			}
		}
	}

}

std::vector<std::vector<std::vector<double>>> NeuralNetwork::getWeightsErrorGradient(std::vector<double> &inputVector, std::vector<double> &outputVector) {
	/*
	Utility function that computes the gradient of error for each weight in our NN.
	i.e, it shows how much each node-to-node weight effects the output error.
	This is used to determine how each node-to-node weight should be changed to reduce the error.
	*/
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
	std::vector<std::vector<double>> networkState = FeedForwardFromInputVector(inputVector);
	y_predicted = networkState[networkState.size() - 1];

	//Use this output and backpropogate to determine the updates we should
	//apply to the weights.Start first from the output layer.

	//Find gradient vector at output nodes
	std::vector<double> delta(y_predicted.size());
	for (int i = 0; i < y_predicted.size(); i++) {
		delta[i] = -2 * (outputVector[i] - y_predicted[i])*gFun_prime(networkState[networkState.size() - 2], nodeWeights[nodeWeights.size() - 1][i]);
		//Add this gradient to our total for the data set and add regularization cost.
		int prevVectorSize = networkState[networkState.size() - 2].size();
		for (int j = 0; j < prevVectorSize; j++) {
			gradient_sum[NumLayers()-2][i][j] = gradient_sum[NumLayers() - 2][i][j] + delta[i] * networkState[networkState.size() - 2][j]
				+ REGULARIZATION_WEIGHT * nodeWeights[nodeWeights.size() - 1][i][j];
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

int NeuralNetwork::NumLayers() {
	//Utility function to return the number of layers in the NN, including input and output layers.
	return nodesPerLayer.size();
}

std::vector<std::vector<double>> NeuralNetwork::FeedForwardFromInputVector(std::vector<double> &inputVector)
{	
	/*
	Computes the value at each node of the NN when a given inputVector is input.
	Includes the constant '1.0's in the hidden layer that make up the 'augmented NN'
	*/
	if (inputVector.size() != nodesPerLayer[0]) throw std::invalid_argument("ERROR: input vector size and NN 1st layer size must match");
	std::vector<std::vector<double>> newNodeValues(NumLayers());
	newNodeValues[0] = inputVector;
	newNodeValues[0].push_back(1.0); //augment the resulting vector with a constant 1.0 to allow a linear offset term.
	std::vector<double> currentValues = newNodeValues[0];

	for (int layer = 1; layer < NumLayers() - 1; layer++){
		//The last layer uses a different multiplier function at each node to get the output value, 
		//so we'll stop this loop before reaching that last layer
		newNodeValues[layer] = std::vector<double>(nodesPerLayer[layer]);
		for (int calcNode = 0; calcNode < nodesPerLayer[layer]; calcNode++){
			newNodeValues[layer][calcNode] = sigmoidFunc(currentValues, nodeWeights[layer-1][calcNode]);
		}
		newNodeValues[layer].push_back(1.0); //augment the resulting vector with a constant 1.0 to allow a linear offset term.
		currentValues = newNodeValues[layer];
	}

	//Calculate values of last network layer by applying gFunction to values at second to last network layer.
	newNodeValues[NumLayers() - 1] = std::vector<double> (nodesPerLayer[NumLayers()-1]);
	for (int calcNode = 0; calcNode < nodesPerLayer[NumLayers() - 1]; calcNode++) {
		newNodeValues[NumLayers() - 1][calcNode] = gFun(currentValues, nodeWeights[NumLayers() - 2][calcNode]);
	}

	return newNodeValues;
}


double NeuralNetwork::sigmoidFunc(std::vector<double> &x, std::vector<double> &weights) {
	//Activation function applied at each hidden node

	if (x.size() != weights.size()) throw std::invalid_argument("ERROR: input vector and weights must be same length");

	double dotProd = 0;
	for (int i = 0; i < x.size(); i++) dotProd += x[i] * weights[i];
	double outVal = 1 / (1 + exp(-dotProd));
	return outVal;

}

double NeuralNetwork::sigmoidFunc_deriv(std::vector<double> &x, std::vector<double> &weights) {
	//Derivative of the activation function applied at each hidden node
	if (x.size() != weights.size()) throw std::invalid_argument("ERROR: input vector and weights must be same length");

	double dotProd = 0;
	for (int i = 0; i < x.size(); i++) dotProd += x[i] * weights[i];
	double expOfMinusDotProd = exp(-dotProd);
	return expOfMinusDotProd / pow((1 + expOfMinusDotProd), 2.0);
}

double NeuralNetwork::gFun(std::vector<double> &x, std::vector<double> &weights) {
	//Activation function applied at each output node, specialized for binary classification
	//This function is specialized for binary classification. It tends to input +/- 1, and REQUIRES TRAINING DATA WHERE THE OUTPUT IS ALWAYS +/-1, otherwise the NN will not predict correctly
	if (x.size() != weights.size()) throw std::invalid_argument("ERROR: input vector and weights must be same length");

	double dotProd = 0;
	for (int i = 0; i < x.size(); i++) dotProd += x[i] * weights[i];
	double expOfMinusDotProd = exp(-dotProd);

	return (1 - expOfMinusDotProd) / (1 + expOfMinusDotProd);
}

double NeuralNetwork::gFun_prime(std::vector<double> &x, std::vector<double> &weights) {
	//Derivative of the activation function applied at each output node, specialized for binary classification
	//This function is specialized for binary classification. It tends to input +/- 1, and REQUIRES TRAINING DATA WHERE THE OUTPUT IS ALWAYS +/-1, otherwise the NN will not predict correctly
	if (x.size() != weights.size()) throw std::invalid_argument("ERROR: input vector and weights must be same length");

	double dotProd = 0;
	for (int i = 0; i < x.size(); i++) dotProd += x[i] * weights[i];
	double expOfMinusDotProd = exp(-dotProd);

	return (2.0 * expOfMinusDotProd) / pow((1 + expOfMinusDotProd),2.0);
}

