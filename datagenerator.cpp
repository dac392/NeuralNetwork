#include "datagenerator.h"
#include <random>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <map> // Include for std::map
#include <set>
#include <vector>

#define RED 2
#define BLUE 3
#define YELLOW 4
#define GREEN 5

DataGenerator::DataGenerator():dangerous_only(false){}

void DataGenerator::toggle(){
    dangerous_only = true;
}

bool DataGenerator::randomBool() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    return dis(gen) == 1;
}

int DataGenerator::randomIndex(std::set<int> &usedIndices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 19);
    int index;

    do {
        index = dis(gen);
    } while (usedIndices.find(index) != usedIndices.end());

    usedIndices.insert(index);
    return index;
}

void DataGenerator::fillDiagram(std::vector<std::vector<std::string>> &diagram, bool isRow, int index, const std::string &color) {
    for (int i = 0; i < 20; ++i) {
        if (isRow) {
            diagram[index][i] = color[0];
        } else {
            diagram[i][index] = color[0];
        }
    }
}

std::string DataGenerator::diagramToString(const std::vector<std::vector<std::string>>& diagram) {
    std::string result;
    for (const auto &row : diagram) {
        for (const auto &cell : row) {
            result += "[" + cell + "]";
        }
        result += "\n";
    }
    return result;
}

std::vector<std::string> DataGenerator::generateDiagram() {
    std::vector<std::vector<std::string>> diagram(20, std::vector<std::string>(20, " "));
    std::vector<std::string> colors = {"Red", "Blue", "Yellow", "Green"};
    std::set<int> usedRows, usedCols;

    // Randomly shuffle the colors
    std::random_device rd;
    std::mt19937 g(rd());
    if(dangerous_only){
        std::shuffle(colors.begin() + 2, colors.end(), g);
    }else{
        std::shuffle(colors.begin(), colors.end(), g);
    }

    bool isRow = randomBool();
    bool redBeforeYellow = false;
    bool redFound = false;

    std::vector<std::string> colors_used;
    for (const auto &color : colors) {
        colors_used.push_back(color);
        if (color == "Red") {
            redFound = true;
        } else if (color == "Yellow" && redFound) {
            redBeforeYellow = true;
        }

        int index = randomIndex(isRow ? usedRows : usedCols);
        fillDiagram(diagram, isRow, index, color);

        isRow = !isRow; // Toggle between row and column
    }

    // Compute additional features using the filled diagram
    std::string status = redBeforeYellow ? "Dangerous" : "Safe";
    std::string thirdColor = colors_used[2];
    return {diagramToString(diagram), status, thirdColor};
}



void DataGenerator::generateDataset(int n_train, int n_test, Eigen::MatrixXd& X_train, Eigen::VectorXd& y_train, Eigen::MatrixXd& Z_train, Eigen::MatrixXd& X_test, Eigen::VectorXd& y_test, Eigen::MatrixXd& Z_test) {
    int num_one_hot_features = 1601; // Number of features from one-hot encoding
    int num_extra_features = 324;
    int total_features = num_one_hot_features + num_extra_features;

    if(!dangerous_only){
        // Resize matrices to accommodate both one-hot encoded and additional features
        X_train.resize(n_train, total_features);
        y_train.resize(n_train);
        Z_train.resize(n_train, 4);
        X_test.resize(n_test, total_features);
        y_test.resize(n_test);
        Z_test.resize(n_test, 4);     
    }


    for (int i = 0; i < n_train; ++i) {
        auto diagramInfo = generateDiagram();
        Eigen::VectorXd encoded = diagramToOneHot(diagramInfo[0]);
        Eigen::VectorXd quadratic = computeLocalQuadraticFeatures(oneHotToMatrix(encoded));
        if (encoded.size() != num_one_hot_features - 1) {
            throw std::runtime_error("Encoded diagram has incorrect size.");
        }

        // Combine one-hot encoded features with additional features
        X_train.row(i) << 1 , encoded.transpose(), quadratic.transpose();
        y_train(i) = diagramInfo[1] == "Dangerous" ? 1 : 0;
        Z_train.row(i) << colorToOneHot(diagramInfo[2]).transpose();

    }
    for (int i = 0; i < n_test; ++i) {
        auto diagramInfo = generateDiagram();
        Eigen::VectorXd encoded = diagramToOneHot(diagramInfo[0]);
        Eigen::VectorXd quadratic = computeLocalQuadraticFeatures(oneHotToMatrix(encoded));
        if (encoded.size() != num_one_hot_features - 1) {
            throw std::runtime_error("X_test Encoded diagram has incorrect size.");
        }

        // Combine one-hot encoded features with additional features
        X_test.row(i) << 1, encoded.transpose(), quadratic.transpose();
        y_test(i) = diagramInfo[1] == "Dangerous" ? 1 : 0;
        Z_test.row(i) << colorToOneHot(diagramInfo[2]).transpose();
    }
}


Eigen::VectorXd DataGenerator::diagramToOneHot(const std::string& diagramString) {
    Eigen::VectorXd oneHot(1600); // 20x20 * 4 + 1 = 1601
    // oneHot(0) = 1; // First element is always 1
    int index = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.05, 0.05);

    std::istringstream stream(diagramString);
    std::string line;

    while (std::getline(stream, line)) {
        for (char ch : line) {
            if (ch == 'R' || ch == 'B' || ch == 'Y' || ch == 'G') {
                // Assign one-hot vector based on color
                oneHot.segment<4>(index) = (ch == 'R') ? Eigen::Vector4d(1, 0, 0, 0) :
                                           (ch == 'B') ? Eigen::Vector4d(0, 1, 0, 0) :
                                           (ch == 'Y') ? Eigen::Vector4d(0, 0, 1, 0) :
                                                         Eigen::Vector4d(0, 0, 0, 1);
                index += 4;
            } else if (ch == ' ') {
                // Assign random noise for empty cells
                oneHot.segment<4>(index) << dis(gen), dis(gen), dis(gen), dis(gen);
                // oneHot.segment<4>(index) << 0, 0, 0, 0;
                index += 4;
            }
        }
    }
    return oneHot;
}

Eigen::Vector4d DataGenerator::colorToOneHot(const std::string& color) {
    if (color == "Red") {
        return Eigen::Vector4d(1, 0, 0, 0);
    } else if (color == "Blue") {
        return Eigen::Vector4d(0, 1, 0, 0);
    } else if (color == "Yellow") {
        return Eigen::Vector4d(0, 0, 1, 0);
    } else if (color == "Green") {
        return Eigen::Vector4d(0, 0, 0, 1);
    } else {
        throw std::invalid_argument("Invalid color for one-hot encoding.");
    }
}

int decode(const Eigen::Vector4d& vec){
    if (vec == Eigen::Vector4d(1, 0, 0, 0)) {
        return RED;
    } else if (vec == Eigen::Vector4d(0, 1, 0, 0)) {
        return BLUE;
    } else if (vec == Eigen::Vector4d(0, 0, 1, 0)) {
        return YELLOW;
    } else if (vec == Eigen::Vector4d(0, 0, 0, 1)) {
        return GREEN;
    }
    return 0;
}

Eigen::MatrixXd DataGenerator::oneHotToMatrix(const Eigen::VectorXd& oneHotVector) {
    if (oneHotVector.size() != 1600) {
        throw std::runtime_error("Invalid oneHotVector size: Expected size 1600");
    }

    Eigen::MatrixXd matrix(20, 20);
    for (int i = 0; i < oneHotVector.size(); i += 4) {
        int row = (i / 4) / 20; // Divide by 4 because of the one-hot encoding
        int col = (i / 4) % 20; // Divide by 4 because of the one-hot encoding
        // Find the index of the max element in the 4-length segment
        Eigen::Vector4d comp;
        for (int j = 0; j < 4; ++j) {
            comp(j) = oneHotVector(i + j);
            
        }

        matrix(row, col) = decode(comp);
    }

    return matrix;
}

Eigen::Vector4d DataGenerator::getRepresentation(int i){
    if(i == RED){
        return Eigen::Vector4d(1, 0, 0, 0);
    }else if(i == BLUE){
        return Eigen::Vector4d(0, 1, 0, 0);
    }else if(i == YELLOW){
        return Eigen::Vector4d(0, 0, 1, 0);
    }else if(i == GREEN){
        return Eigen::Vector4d(0, 0, 0, 1);
    }

    return Eigen::Vector4d(0, 0, 0, 0);
}

Eigen::VectorXd DataGenerator::computeLocalQuadraticFeatures(const Eigen::MatrixXd& matrix) {
    // Eigen::MatrixXd quadraticFeatures(20, 20);
    // Eigen::MatrixXd quadraticFeatures(9, 9);
    // std::vector<double> quadraticFeatures;
    Eigen::VectorXd quadraticFeatures(324);
    int stepSize = 2;
    int kernel_size = 4;
    int index = 0;
    for (int row = 0; row < 20 - 2; row+=stepSize) {
        for (int col = 0; col < 20 - 2; col+=stepSize) {
            Eigen::Vector4d encoded(0, 0, 0, 0);
            // Check and multiply with neighbors: up, right, down, left
            // if (row > 0) quadraticFeatures.push_back(cellValue * cellValue * matrix(row - 1, col)); // Up
            // if (col < matrix.cols() - 1) quadraticFeatures.push_back(cellValue * cellValue * matrix(row, col + 1)); // Right
            // if (row < matrix.rows() - 1) quadraticFeatures.push_back(cellValue * cellValue * matrix(row + 1, col)); // Down
            // if (col > 0) quadraticFeatures.push_back(cellValue * cellValue * matrix(row, col - 1)); // Left

            for(int i = 0; i < kernel_size; i++){
                for(int j = 0; j < kernel_size; j++){
                    encoded += getRepresentation(matrix(row+i, col+j));
                }
            }
            for(int k = 0; k < 4; k++){
                quadraticFeatures(index+k) = encoded(k);
            }
            
            index+=4;
        }
    }

    // std::cout << quadraticFeatures.size() << std::endl;
    // return Eigen::Map<Eigen::VectorXd>(quadraticFeatures.data(), quadraticFeatures.size());
    // return Eigen::VectorXd::Map(quadraticFeatures.data(), quadraticFeatures.size());
    return quadraticFeatures;
}




