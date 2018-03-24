#pragma once
#pragma once

#include <string>
#include <sstream>
#include <istream>
#include <vector>

namespace CSVReader {
	static bool ReadLineIntoDoubleVector(std::istream& str, std::vector<double>& dblVector);
	static bool ReadLineIntoIntVector(std::istream& str, std::vector<int>& intVector);
	static bool ReadLineIntoStringVector(std::istream& str, std::vector<std::string>& strVector);
}

namespace CSVReader {


	bool ReadLineIntoStringVector(std::istream& str, std::vector<std::string>& strVector) {
		strVector.clear();
		std::string line;
		try {
			std::getline(str, line);

			std::stringstream lineStream(line);
			std::string cell;

			while (std::getline(lineStream, cell, ',')) {
				strVector.push_back(cell);
			}

			if (!lineStream && cell.empty()) {
				strVector.push_back("");
			}
		}
		catch (std::exception &e) {	
			auto errorString = e.what();
			return false;
		}

		return true;

	}

	bool ReadLineIntoIntVector(std::istream& str, std::vector<int>& intVector) {
		intVector.clear();
		std::vector<std::string> strVector;
		bool success = ReadLineIntoStringVector(str, strVector);
		if (success) {
			for (int i = 0; i < (int)(strVector.size()); i++) {
				try {
					intVector.push_back(std::stoi(strVector[i]));
				}
				catch (std::exception e) {
					intVector.clear();
					success = false;
					break;
				}
			}
		}
		return success;
	}

	bool ReadLineIntoDoubleVector(std::istream& str, std::vector<double>& dblVector) {
		dblVector.clear();
		std::vector<std::string> strVector;
		bool success = ReadLineIntoStringVector(str, strVector);
		if (success) {
			for (int i = 0; i < (int)(strVector.size()); i++) {
				try {
					dblVector.push_back(std::stod(strVector[i]));
				}
				catch (std::exception e) {
					dblVector.clear();
					success = false;
					break;
				}
			}
		}
		return success;
	}
}