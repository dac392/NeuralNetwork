#include "logisticregression.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip> // for std::setw
#include <random>
#include <cmath>

#define THRESH 0.55
// chcp 65001

LogisticRegression::LogisticRegression(double lr, int iter, double regStrength, RegularizationType regType)
    : learningRate(lr), iterations(iter), regularizationStrength(regStrength), regType(regType) {
        datasize = 0;
}

// Initialize weights to small random values or zeros
void LogisticRegression::initializeWeights(int n_features) {
    weights = Eigen::VectorXd::Random(n_features); // uniform [-1,1]
    weights = weights * 0.015;

    fixing_weights = Eigen::MatrixXd::Random(n_features, 4); // uniform [-1,1]
    fixing_weights = fixing_weights * 0.015;

    std::cout << "Weights: " <<  weights.rows() << " x " << weights.cols() << std::endl;
    std::cout << "Fixing_weights: " <<  fixing_weights.rows() << " x " << fixing_weights.cols() << std::endl;
}

void LogisticRegression::setDatasize(int i){
    datasize = i;
}


// Fit the model using gradient descent
void LogisticRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& X_test, const Eigen::VectorXd& y_test) {
    initializeWeights(X.cols());
    gradientDescent(X, y, X_test, y_test);
    // SGD(X, y, X_test, y_test);
    double training_accuracy = predict(X, y, THRESH);
    double testing_accuracy = predict(X_test, y_test, THRESH);
    std::cout << "Training Accuracy: " << training_accuracy
              << ", Testing Accuracy: " << testing_accuracy
              << ", LearningRate: " << learningRate
              << ", Lambda: " << regularizationStrength
              << ", and Threshold: " << THRESH << std::endl;

    // Append to the file
    std::ofstream file("accuracy_v_datasize_fit.txt", std::ios::app); // Open in append mode
    if (file.is_open()) {
        file << "TrainingAccuracy: " << training_accuracy << ", "
             << "TestingAccuracy: " << testing_accuracy << ", "
             << "DataSize: " << datasize << std::endl;
        file.close();
    } else {
        std::cerr << "Unable to open file." << std::endl;
    }
}

// calculates accuracy
double LogisticRegression::predict(const Eigen::MatrixXd& X_test, const Eigen::VectorXd& y_test, double threshold) {
    Eigen::VectorXd predictions = (X_test * weights).unaryExpr(&sigmoid);
    predictions = (predictions.array() > threshold).cast<double>(); // Convert probabilities to 0 or 1

    // Calculate accuracy
    double correct = (predictions.array() == y_test.array()).cast<double>().sum();
    double accuracy = correct / y_test.size();

    return accuracy;
}

// stochastic gradient descent
void LogisticRegression::SGD(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& X_test, const Eigen::VectorXd& y_test){
    
    int n_samples = X.rows();
    int batch_size = 25;
    int n_batches = n_samples / batch_size;

    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., n_samples - 1
    std::ofstream loss_file("loss_data.txt");
    int progressBarWidth = 50; // Width of the progress bar
    for (int iter = 0; iter < iterations/5; ++iter) {
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine());

        for (int b = 0; b < n_batches; ++b) {
            std::vector<int> batch_indices(indices.begin() + b * batch_size, 
                                           indices.begin() + (b + 1) * batch_size);

            // Extract the mini-batch from X and y
            Eigen::MatrixXd X_batch(batch_size, X.cols());
            Eigen::VectorXd y_batch(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                X_batch.row(i) = X.row(batch_indices[i]);
                y_batch[i] = y[batch_indices[i]];
            }


            // Compute gradient and update weights
            Eigen::VectorXd gradient = computeGradient(X_batch, y_batch);
            weights -= learningRate * gradient;
        }

        Eigen::VectorXd predictions = predict_probabilities(X);
        double train_loss = computeLoss(y, predictions);
        double test_loss = computeLoss(y_test, predict_probabilities(X_test));   // Compute current testing loss
        loss_file << iter << " " << train_loss << " " << test_loss << std::endl;

        // Update the progress bar
        double progress = (iter + 1) / static_cast<double>(iterations/5);
        std::cout << "[";
        int pos = progressBarWidth * progress;
        for (int i = 0; i < progressBarWidth; ++i) {
            if (i < pos) std::cout << "█";  // Block character for solid fill
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();

    }
    loss_file.close();
    std::cout << "SGD - ";
}

// Gradient Descent
void LogisticRegression::gradientDescent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& X_test, const Eigen::VectorXd& y_test){
    int n_samples = X.rows();
    std::ofstream loss_file("loss_data.txt");

    int progressBarWidth = 50; // Width of the progress bar
    for (int epoch = 0; epoch < iterations; ++epoch) {
        Eigen::VectorXd predictions = predict_probabilities(X);
        Eigen::VectorXd gradient = X.transpose() * (predictions - y) / n_samples;

        if (regType == RegularizationType::L2) {
            Eigen::VectorXd weights_without_bias = weights;
            weights_without_bias(0) = 0; // Exclude bias from regularization if it's the first element
            gradient += regularizationStrength * weights_without_bias;
        }

        weights -= learningRate * gradient;

        double train_loss = computeLoss(y, predictions);
        double test_loss = computeLoss(y_test, predict_probabilities(X_test));   // Compute current testing loss
        loss_file << epoch << " " << train_loss << " " << test_loss << std::endl;
        // std::cout << "Epoch " << epoch << ", Training_Loss: " << train_loss << ", Testing_Loss: " << test_loss <<std::endl;

        // Update the progress bar
        double progress = (epoch + 1) / static_cast<double>(iterations);
        std::cout << "[";
        int pos = progressBarWidth * progress;
        for (int i = 0; i < progressBarWidth; ++i) {
            if (i < pos) std::cout << "█";  // Block character for solid fill
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();

    }

    loss_file.close();
    std::cout << "gradient descent - ";
}

// Compute Gradient
Eigen::VectorXd LogisticRegression::computeGradient(const Eigen::MatrixXd& X_batch, const Eigen::VectorXd& y_batch) { 
    int n_samples = X_batch.rows();
    // Predict probabilities for the mini-batch
    Eigen::VectorXd predictions = (X_batch * weights).unaryExpr(&sigmoid);  
    Eigen::VectorXd gradient = (X_batch.transpose() * (predictions - y_batch)) / n_samples;

    // Add regularization term if applicable
    if (regType == RegularizationType::L1) {
        // L1 regularization: add the sign of weights
        Eigen::VectorXd sign_weights = weights.unaryExpr([](double w) { return (w > 0) ? 1.0 : (w < 0) ? -1.0 : 0.0; });
        gradient += regularizationStrength * sign_weights / n_samples;
    } else if (regType == RegularizationType::L2) {
        // L2 regularization: add the weights themselves
        gradient += regularizationStrength * weights / n_samples;
    }

    return gradient;
}

// Compute binary cross-entropy loss
double LogisticRegression::computeLoss(const Eigen::VectorXd& y, const Eigen::VectorXd& predictions) const {
    int n_samples = y.size();
    double loss = -(y.array() * (predictions.array() + 1e-10).log() 
                  + (1 - y.array()) * (1 - predictions.array() + 1e-10).log()).sum();

    // Add regularization term
    if (regType == RegularizationType::L1) {
        loss += regularizationStrength * weights.cwiseAbs().sum() / n_samples;
    } else if (regType == RegularizationType::L2) {
        loss += 0.5 * regularizationStrength * weights.squaredNorm() / n_samples;
    }

    return loss / n_samples;
}

// Predict probabilities using the logistic function
Eigen::VectorXd LogisticRegression::predict_probabilities(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd predictions = (X * weights).unaryExpr(&sigmoid);
    return predictions;
}

double LogisticRegression::sigmoid(double z) {
    // Clipping z to avoid overflow in exp(-z)
    double clipped_z = std::max(-20.0, std::min(20.0, z));
    return 1.0 / (1.0 + exp(-clipped_z));
}

void LogisticRegression::setLearningRate(double lr){
    learningRate = lr;
}
void LogisticRegression::setRegularizationStrength(double reg){
    regularizationStrength = reg;
}








void LogisticRegression::fitMore(const Eigen::MatrixXd& X, const Eigen::MatrixXd& z, const Eigen::MatrixXd& X_test, const Eigen::MatrixXd& z_test){
    SGD2(X, z, X_test, z_test);
    std::cout << "SGD2 - ";
    double training_accuracy = predict2(X, z);
    double testing_accuracy = predict2(X_test, z_test);
    std::cout << "Training Accuracy: " << training_accuracy
              << ", Testing Accuracy: " << testing_accuracy
              << ", LearningRate: " << learningRate
              << ", Lambda: " << regularizationStrength << std::endl;

    // Append to the file
    std::ofstream file("accuracy_v_datasize_fitMore.txt", std::ios::app); // Open in append mode
    if (file.is_open()) {
        file << "TrainingAccuracy: " << training_accuracy << ", "
             << "TestingAccuracy: " << testing_accuracy << ", "
             << "DataSize: " << datasize << std::endl;
        file.close();
    } else {
        std::cerr << "Unable to open file." << std::endl;
    }
}

void LogisticRegression::SGD2(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& X_test, const Eigen::MatrixXd& Z_test) {
    int n_samples = X.rows();
    int batch_size = 25;
    int n_batches = n_samples / batch_size;
    int progressBarWidth = 50; // Width of the progress bar
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    std::ofstream loss_file("task2_loss_data.txt");
    for (int iter = 0; iter < iterations/2; ++iter) {
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine());

        for (int b = 0; b < n_batches; ++b) {
            std::vector<int> batch_indices(indices.begin() + b * batch_size, 
                                           indices.begin() + (b + 1) * batch_size);

            Eigen::MatrixXd X_batch(batch_size, X.cols());
            Eigen::MatrixXd Z_batch(batch_size, 4); // 4 classes for 4 wires
            for (int i = 0; i < batch_size; ++i) {

                X_batch.row(i) = X.row(batch_indices[i]);
                Z_batch.row(i) = Z.row(batch_indices[i]);
            }

            Eigen::MatrixXd gradient = softmaxGradient(X_batch, Z_batch);
            fixing_weights -= learningRate * gradient;
        }

        double train_loss = computeSecondLoss(Z, softmax(X*fixing_weights));
        double test_loss = computeSecondLoss(Z_test, softmax(X_test*fixing_weights));   // Compute current testing loss
        loss_file << iter << " " << train_loss << " " << test_loss << std::endl;

        // Update the progress bar
        double progress = (iter + 1) / static_cast<double>(iterations/5);
        std::cout << "[";
        int pos = progressBarWidth * progress;
        for (int i = 0; i < progressBarWidth; ++i) {
            if (i < pos) std::cout << "█";  // Block character for solid fill
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();

    }
}

// One-hot encode the predictions of softmax
void oneHotEncode(Eigen::MatrixXd& probabilities) {
    for (int i = 0; i < probabilities.rows(); ++i) {
        int maxIndex = 0;
        double maxValue = probabilities(i, 0);
        
        // Find the index of the max value in the row
        for (int j = 1; j < probabilities.cols(); ++j) {
            if (probabilities(i, j) > maxValue) {
                maxValue = probabilities(i, j);
                maxIndex = j;
            }
        }

        // Set all values to 0 and the max value to 1
        probabilities.row(i).setZero();
        probabilities(i, maxIndex) = 1;
    }
}

// calculate accuracy
double LogisticRegression::predict2(const Eigen::MatrixXd& X_test, const Eigen::MatrixXd& y_test) {
    Eigen::MatrixXd z = X_test * fixing_weights; // Compute raw predictions (logits)
    Eigen::MatrixXd predictions = softmax(z); // Apply softmax to get probabilities
    oneHotEncode(predictions);

    // Calculate accuracy
    double correct = 0;
    for (int i = 0; i < y_test.rows(); ++i) {
        if (predictions(i) == y_test(i)) { // Use predictions(i) instead of predictedClass
            ++correct;
        }
    }
    double accuracy = correct / y_test.rows();

    return accuracy;
}


// Compute Gradient for Softmax
Eigen::MatrixXd LogisticRegression::softmaxGradient(const Eigen::MatrixXd& X_batch, const Eigen::MatrixXd& Z_batch) {
    int n_samples = X_batch.rows();
    Eigen::MatrixXd predictions = softmax(X_batch * fixing_weights);

    // Gradient of the cross-entropy loss
    Eigen::MatrixXd error = predictions - Z_batch; // Error term
    Eigen::MatrixXd gradient = (X_batch.transpose() * error) / n_samples;

    if (regType == RegularizationType::L2) {
        gradient += regularizationStrength * fixing_weights / n_samples;
    }

    return gradient;
}

Eigen::MatrixXd LogisticRegression::softmax(const Eigen::MatrixXd& Z) {
    Eigen::MatrixXd expZ = Z.unaryExpr([](double z) { return std::exp(z); });
    Eigen::VectorXd sumExpZ = expZ.rowwise().sum();

    for (int i = 0; i < Z.rows(); ++i) {
        expZ.row(i) /= sumExpZ(i);
    }

    return expZ;
}

// Compute binary cross-entropy loss for Softmax
double LogisticRegression::computeSecondLoss(const Eigen::MatrixXd& y, const Eigen::MatrixXd& predictions) const {
    int n_samples = y.rows();

    // Compute the cross-entropy loss
    double loss = -(y.array() * (predictions.array() + 1e-10).log()).sum();

    // Add regularization term
    if (regType == RegularizationType::L2) {
        loss += 0.5 * regularizationStrength * fixing_weights.squaredNorm() / n_samples;
    }

    return loss / n_samples;
}



