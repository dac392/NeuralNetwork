#include "logisticregression.h"
#include "datagenerator.h"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <cmath>

#define LEARNING_RATE 0.1
#define LAMBDA 0.0001

#define ITERATIONS 5000

#define N_TRAINING 5000
#define N_TESTING 1000

// loss.txt - loss over time
// datasizes.txt - accuracy v. data sizes
// accuracy.txt - accuracy v. iterations
// learning rates - accuracy v. learning rates
// lambda.txt - lambda v. accuracy
// threshold v. accuracy


void printOneHotEncoded(const Eigen::VectorXd& vector) {
    std::ostringstream output;
	output << static_cast<int>(vector(0)) << "\n";
    for (int i = 1; i < vector.size(); ++i) {
        output << static_cast<int>(vector(i)); // Print each element as an integer

        if ((i) % 4 == 0) {
            output << " "; // Add a space every 4 elements
        }
        if ((i) % 80 == 0) {
            output << "\n"; // Add a new line every 80 elements (20x4)
        }
    }

    std::cout << output.str() << std::endl;
}

void shapeOf(const Eigen::MatrixXd& X){
	std::cout << "shape of X_train( " << X.rows() << ", " << X.cols() << " )" << std::endl;
}

// Collect indices where y_train and y_test are 1
void collectIndecies(std::vector<int>& train_indices, std::vector<int>& test_indices, const Eigen::VectorXd& y_train, const Eigen::VectorXd& y_test){
	
	for (int i = 0; i < y_train.size(); ++i) {
	    if (y_train[i] == 1) {
	        train_indices.push_back(i);
	    }
	}

	for (int i = 0; i < y_test.size(); ++i) {
	    if (y_test[i] == 1) {
	        test_indices.push_back(i);
	    }
	}
}

int main(){
	//2k, 2.5k, 3k, 5k

	Eigen::MatrixXd X_train;
	Eigen::MatrixXd X_test;

	Eigen::VectorXd y_train;
	Eigen::VectorXd y_test;

	Eigen::MatrixXd Z_train;
	Eigen::MatrixXd Z_test;

	Eigen::MatrixXd A_train, B_train, A_test, B_test;

	std::vector<float> learning_rates = {0.1, 0.01, 0.001, 0.0001};
	std::vector<float> lambda_rates = {0.0001, 0.1, 0.01, 0.001, 0.00001, 0.000001};
	std::vector<float> traning_data = {2000, 2500, 3000, 5000};

	for(const auto& training : traning_data){

		DataGenerator generator = DataGenerator();
		generator.generateDataset(training, training/5, X_train, y_train, Z_train, X_test, y_test, Z_test);

		LogisticRegression model(LEARNING_RATE, ITERATIONS, LAMBDA, RegularizationType::L2);
		model.setDatasize(training);
		model.fit(X_train, y_train, X_test, y_test);

		std::vector<int> train_indices, test_indices;
		collectIndecies(train_indices, test_indices, y_train, y_test);

		// Resize A_train, B_train, A_test, B_test matrices
		A_train.resize(train_indices.size(), X_train.cols());
		B_train.resize(train_indices.size(), Z_train.cols());
		A_test.resize(test_indices.size(), X_test.cols());
		B_test.resize(test_indices.size(), Z_test.cols());

		// Fill A_train, B_train, A_test, B_test
		for (size_t i = 0; i < train_indices.size(); ++i) {
		    A_train.row(i) = X_train.row(train_indices[i]);
		    B_train.row(i) = Z_train.row(train_indices[i]);
		}

		for (size_t i = 0; i < test_indices.size(); ++i) {
		    A_test.row(i) = X_test.row(test_indices[i]);
		    B_test.row(i) = Z_test.row(test_indices[i]);
		}

		model.fitMore(A_train, B_train, A_test, B_test);	
	}

	



	return 0;
}



