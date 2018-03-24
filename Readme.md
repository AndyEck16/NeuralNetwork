# Neural Network Demo

## Overview
 This project is a demo for a Neural Network class. Allows a user to
  * Construct a Neural network of arbitrary number of layers and nodes per layer
  * Set activation functions at the nodes in the network. (Output layer nodes may have different activation functions than hidden layer nodes.)
  * Train a constructed network from a data file
  * Test a trained network against a data file
  
## Testing and Verification 
  
This class was tested using the data files in the *TrainingData* folder using a NN with the following properties:

* Output layer activation function: NNUtilities::binaryClassifierActivFunc
* Hidden layer activation function: NNUtilities::sigmoidFunc 
* Learning_Step_Size = 0.001 
* Regularization_Weight = 0.001 
* nodesPerLayer = ({2, 5, 4, 1})

After training with *nandData.csv*, these settings should get perfect prediction accuracy on *nandTestData.csv*.
  
## Summary of files and classes in this repository

### **Folder: CSVReader**

  #### *CSVReader (class)* 
  * This folder contains a utility file that reads data from a CSV file and puts it into vector containers
    * Note: expects training data formatted as CSV containing only ',' and newline / blankspace characters
    * Note: Each line represents a combined input and output vector. 
           * For example, an input vector "[0 1 1 0]" that produces output vector "[-1 1]" should be represented by the line:
             "0,1,1,0,-1,1"*
  
### Folder: **NeuralNetworkFramework**

  #### *NNUtilityFunctions (namespace)*
  * Activation functions (linear, sigmoid, binary classifier)
  * Inner Product calculator
  * Parse a training vector into the input and output vectors that are contained within it
  * Random Number Generator from a Normal Distribution (mean 0.0, stddev 1.0)
  #### *NeuralNetworkFramework (main)*
  * A sample implementation of the NeuralNetwork class. 
    * Initializes structure of an NN
    * Sets the activation functions at the output layer and hidden layers
    * Trains the network on a sample data file
    * Tests the network against a test data set file.
  #### *NeuralNetwork (class)*
  * User access to the neural network occurs through this class.
  * The public members / methods in this class should be enough for full use of the NN.
  * Private access to a vector of NeuralLayers that make up the network
  #### *NeuralLayer (class)*
  * Each instance of NeuralLayer contains a vector of Neural Nodes
  * Feedforward and backpropogation algorithms for prediction and learning from training data
  * Getters and Setters for node values
  * Pointers to neighboring NeuralLayers
  #### *NeuralNode (class)*
  * Weight vector and bias
  * Activation functions 
  
### Folder: TrainingData
  * Contains two data files representing input and output of a NAND gate. 2D input vector, 1D output vector. 
    * *Example: [0 0] -> [1]; [0 1] -> [1]; [1 0] -> [1]; [1 1] -> [0]*
  * *nandData.csv* contains 80,000 data vectors
  * *nandTestData.csv* contains 20,000 data vectors.
  
