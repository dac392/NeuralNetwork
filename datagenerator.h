#ifndef DATAGENERATOR_H
#define DATAGENERATOR_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <set>

struct VectorElementComparator {
    bool operator() (const Eigen::VectorXd& a, const Eigen::VectorXd& b) const {
        if (a.size() != b.size()) return true;  // Different sizes, so definitely different

        // Compare elements
        for (int i = 0; i < a.size(); ++i) {
            if (a(i) != b(i)) return a(i) < b(i);
        }

        return false;  // All elements are equal
    }
};

class DataGenerator {
public:
    DataGenerator();
    std::vector<std::string> generateDiagram();
    void generateDataset(int n_train, int n_test, Eigen::MatrixXd& X_train, Eigen::VectorXd& y_train, Eigen::MatrixXd& Z_train, Eigen::MatrixXd& X_test, Eigen::VectorXd& y_test, Eigen::MatrixXd& Z_test);
    void toggle();

private:
    bool dangerous_only;

    bool randomBool();
    int randomIndex(std::set<int> &usedIndices);
    void fillDiagram(std::vector<std::vector<std::string>> &diagram, bool isRow, int index, const std::string &color);
    std::string diagramToString(const std::vector<std::vector<std::string>>& diagram);
    Eigen::VectorXd diagramToOneHot(const std::string& diagramString);
    void validateDiagram(const std::string& diagram, const std::string& status);
    Eigen::VectorXd computeAdditionalFeatures(const std::string& diagramString);
    Eigen::Vector4d colorToOneHot(const std::string& color);
    Eigen::MatrixXd oneHotToMatrix(const Eigen::VectorXd& oneHotVector);
    Eigen::VectorXd computeLocalQuadraticFeatures(const Eigen::MatrixXd& matrix);
    Eigen::Vector4d getRepresentation(int i);
};

#endif // DATAGENERATOR_H
