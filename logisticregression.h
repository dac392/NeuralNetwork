#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <Eigen/Dense>
#include <vector>

enum class RegularizationType {
    None,
    L1,
    L2
};

class LogisticRegression {
public:
    LogisticRegression(double learningRate, int iterations, double regularizationStrength, RegularizationType regType = RegularizationType::L2);
    void initializeWeights(int n_features);

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& X_test, const Eigen::VectorXd& y_test);
    void fitMore(const Eigen::MatrixXd& X, const Eigen::MatrixXd& z, const Eigen::MatrixXd& X_test, const Eigen::MatrixXd& z_test);
    void SGD(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& X_test, const Eigen::VectorXd& y_test);
    void SGD2(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& X_test, const Eigen::MatrixXd& Z_test);
    void gradientDescent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& X_test, const Eigen::VectorXd& y_test);
    void classify(const Eigen::MatrixXd& X, const Eigen::MatrixXd& z);

    double predict(const Eigen::MatrixXd& X_test, const Eigen::VectorXd& y_test, double threshold);
    double predict2(const Eigen::MatrixXd& X_test, const Eigen::MatrixXd& y_test);
    Eigen::VectorXd predict_probabilities(const Eigen::MatrixXd& X) const;
    Eigen::VectorXd computeGradient(const Eigen::MatrixXd& X_batch, const Eigen::VectorXd& y_batch);
    Eigen::MatrixXd softmaxGradient(const Eigen::MatrixXd& X_batch, const Eigen::MatrixXd& Y_batch);
    double computeSecondLoss(const Eigen::MatrixXd& y, const Eigen::MatrixXd& predictions) const;

    void setLearningRate(double lr);
    void setRegularizationStrength(double reg);
    void setDatasize(int i);

private:
    double learningRate;
    int iterations;
    double regularizationStrength;
    RegularizationType regType;
    Eigen::VectorXd weights;
    Eigen::MatrixXd fixing_weights;
    int datasize;

    static double sigmoid(double z);
    Eigen::MatrixXd softmax(const Eigen::MatrixXd& Z);
    double computeLoss(const Eigen::VectorXd& y, const Eigen::VectorXd& predictions) const;

};

#endif // LOGISTICREGRESSION_H
