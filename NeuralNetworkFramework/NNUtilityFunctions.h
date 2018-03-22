#ifndef NN_UTILITY_FUNCTIONS_H
#define NN_UTILITY_FUNCTIONS_H

#include <vector>

namespace NNUtilityFunctions {

	static double linearFunc(double inDotProd) {
		return inDotProd;
	}

	static double linearFunc_deriv(double inDotProd) {
		return 1;
	}

	static double sigmoidFunc(double inDotProd) {
		//Activation function applied at each hidden node
		double outVal = 1 / (1 + exp(-inDotProd));
		return outVal;

	}

	static double sigmoidFunc_deriv(double inDotProd) {
		//Derivative of the activation function applied at each hidden node
		double expOfMinusDotProd = exp(-inDotProd);
		return expOfMinusDotProd / pow((1 + expOfMinusDotProd), 2.0);
	}

	static double binaryClassifierActivFunc(double inDotProd) {
		//Activation function applied at each output node, specialized for binary classification
		//This function is specialized for binary classification. It tends to input +/- 1, and REQUIRES TRAINING DATA WHERE THE OUTPUT IS ALWAYS +/-1, otherwise the NN will not predict correctly
		double expOfMinusDotProd = exp(-inDotProd);

		return (1 - expOfMinusDotProd) / (1 + expOfMinusDotProd);
	}

	static double binaryClassifierActivFunc_prime(double inDotProd) {
		//Derivative of the activation function applied at each output node, specialized for binary classification
		//This function is specialized for binary classification. It tends to input +/- 1, and REQUIRES TRAINING DATA WHERE THE OUTPUT IS ALWAYS +/-1, otherwise the NN will not predict correctly
		double expOfMinusDotProd = exp(-inDotProd);

		return (2.0 * expOfMinusDotProd) / pow((1 + expOfMinusDotProd), 2.0);
	}

	static double innerProduct(std::vector<double> const &vec1, std::vector<double> const &vec2) {
		//Return dot product of two vectors
		if (vec1.size() != vec2.size()) throw std::invalid_argument("ERROR: input vector and weights must be same length");

		double dotProd = 0;
		for (int i = 0; i < vec1.size(); i++) dotProd += vec1[i] * vec2[i];

		return dotProd;

	}

	static std::vector<double> GetInputVectorFromCombinedVector(std::vector<double> &combinedVector, int dimensionalityOfInputVector) {
		std::vector<double>::const_iterator inputVectorStart = combinedVector.begin();
		std::vector<double>::const_iterator inputVectorEnd = combinedVector.begin() + dimensionalityOfInputVector;
		std::vector<double> inputVector(inputVectorStart, inputVectorEnd);
		return inputVector;
	}

	static std::vector<double> GetOutputVectorFromCombinedVector(std::vector<double> &combinedVector, int dimensionalityOfInputVector) {
		std::vector<double>::const_iterator outputVectorStart = combinedVector.begin() + dimensionalityOfInputVector;
		std::vector<double>::const_iterator outputVectorEnd = combinedVector.end();
		std::vector<double> outputVector(outputVectorStart, outputVectorEnd);
		return outputVector;
	}

}

#endif