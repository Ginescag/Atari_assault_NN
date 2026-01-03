#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
using namespace std;

class Matrix {
    private:
        vector<vector<double>> data;
        unsigned int rows;
        unsigned int cols;

    public:

    //constructors
        Matrix() : rows(0), cols(0) {}

        Matrix(unsigned int r, unsigned int c) : rows(r), cols(c) {
            data.resize(r, vector<double>(c, 0.0));
        }

        Matrix (const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {}


    //getters
        unsigned int getRows() const {
            return rows;
        }

        unsigned int getCols() const {
            return cols;
        }

        double& at(unsigned int r, unsigned int c) {
            return data[r][c];
        }

        const double& at(unsigned int r, unsigned int c) const {
            return data[r][c];
        }

    //operators
        Matrix& operator=(const Matrix& other) {
            if (this != &other) {
                rows = other.rows;
                cols = other.cols;
                data = other.data;
            }
            return *this;
        }

        Matrix operator*(const Matrix& other) const {
            
            if (cols != other.rows) {
                throw invalid_argument("Matrix dimensions do not match for multiplication.");
            }

            Matrix result(rows, other.cols);
            for (unsigned int i = 0; i < rows; ++i) {
                for (unsigned int j = 0; j < other.cols; ++j) {
                    for (unsigned int k = 0; k < cols; ++k) {
                        result.at(i, j) += at(i, k) * other.at(k, j);
                    }
                }
            }
            return result;
        }

        Matrix operator+(const Matrix& other) const {
            
            if (rows != other.rows || cols != other.cols) {
                throw invalid_argument("Matrix dimensions do not match for addition.");
            }

            Matrix result(rows, cols);
            for (unsigned int i = 0; i < rows; ++i) {
                for (unsigned int j = 0; j < cols; ++j) {
                    result.at(i, j) = at(i, j) + other.at(i, j);
                }
            }
            return result;
        }

        Matrix operator-(const Matrix& other) const {
            
            if (rows != other.rows || cols != other.cols) {
                throw invalid_argument("Matrix dimensions do not match for subtraction.");
            }

            Matrix result(rows, cols);
            for (unsigned int i = 0; i < rows; ++i) {
                for (unsigned int j = 0; j < cols; ++j) {
                    result.at(i, j) = at(i, j) - other.at(i, j);
                }
            }
            return result;
        }

    //auxiliary functions
        Matrix Hadamard(const Matrix& other) const {
            
            if (rows != other.rows || cols != other.cols) {
                throw invalid_argument("Matrix dimensions do not match for Hadamard product.");
            }

            Matrix result(rows, cols);
            for (unsigned int i = 0; i < rows; ++i) {
                for (unsigned int j = 0; j < cols; ++j) {
                    result.at(i, j) = at(i, j) * other.at(i, j);
                }
            }
            return result;
        }

        Matrix map(double (*func)(double)) const {
            Matrix result(rows, cols);
            for (unsigned int i = 0; i < rows; ++i) {
                for (unsigned int j = 0; j < cols; ++j) {
                    result.at(i, j) = func(at(i, j));
                }
            }
            return result;
        }

        Matrix ScalarMul(double scalar) const {
            Matrix result(rows, cols);
            for (unsigned int i = 0; i < rows; ++i) {
                for (unsigned int j = 0; j < cols; ++j) {
                    result.at(i, j) = at(i, j) * scalar;
                }
            }
            return result;
        }

        Matrix transpose() const {
            Matrix result(cols, rows);
            for (unsigned int i = 0; i < rows; ++i) {
                for (unsigned int j = 0; j < cols; ++j) {
                    result.at(j, i) = at(i, j);
                }
            }
            return result;
        }

        void RandMat(){
            for (unsigned int i = 0; i < rows; ++i) {
                for (unsigned int j = 0; j < cols; ++j) {
                    data[i][j] = ((double) rand() / RAND_MAX) * 2 - 1; // Random values between -1 and 1
                }
            }
        }

        void RandIntMat(int minVal, int maxVal){
            for (unsigned int i = 0; i < rows; ++i) {
                for (unsigned int j = 0; j < cols; ++j) {
                    data[i][j] = rand() % (maxVal - minVal + 1) + minVal; // Random integers between minVal and maxVal
                }
            }
        }

        friend ostream& operator<<(ostream& os, const Matrix& mat) {
            for (unsigned int i = 0; i < mat.rows; ++i) {
                for (unsigned int j = 0; j < mat.cols; ++j) {
                    os.width(6);
                    os << mat.at(i, j) << " ";
                }
                os << endl;
            }
            return os;
        }
};


class NeuralNetwork {
    private:
        vector<unsigned int> layers;
        vector<Matrix> weights;
        vector<Matrix> biases;
        vector<Matrix> activations;
    
        static double sigmoid(double x) {
            return 1.0 / (1.0 + exp(-x));
        }

        static double dsigmoid(double y) {
            return y * (1.0 - y);
        }
        
        static double tanh(double x) {
            return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
        }

        // y = tanh(x)
        static double dtanh(double y) {
            return 1.0 - y * y;
        }
        
        static double sign(double x) {
            return (x >= 0) ? 1.0 : -1.0;
        }
    
    public:
        NeuralNetwork(const vector<unsigned int>& layer_sizes) : layers(layer_sizes) {
            for(int i = 0; i < layers.size() - 1; ++i) {
                Matrix weight_matrix(layers[i + 1], layers[i]);
                weight_matrix.RandMat();
                weights.push_back(weight_matrix);

                Matrix bias_matrix(layers[i + 1], 1);
                bias_matrix.RandMat();
                biases.push_back(bias_matrix);
            }
        }

        Matrix feedforward(const Matrix& input, string activation_func = "sigmoid") {
            activations.clear();
            activations.push_back(input);
            Matrix activation = input;

            for(int i = 0; i < weights.size(); ++i) {
                activation = (weights[i] * activation) + biases[i];
                if (activation_func == "sigmoid") {
                    activation = activation.map(sigmoid);
                } else if (activation_func == "tanh") {
                    activation = activation.map(tanh);
                } else {
                    cerr << "Unknown activation function: " << activation_func << endl;
                    exit(-1);
                }

                activations.push_back(activation);
            }
            return activation;
        }


        
};

#endif