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

        //we only implement softmax for the output layer (1 column matrix)
        static Matrix softmax(const Matrix& z){
           //softmax formula -> exp(z_i) / sum(exp(z_j)) for j = 1 to n
           //we need to normalize the output vector to avoid overflow ej (exp(1000) is too large to handle)

            //first we find the max element in the output vector to get the offset for numerical stability
            Matrix result(z.getRows(), z.getCols());
            double maxElem = z.at(0,0);
            for(unsigned int i = 0; i < z.getRows(); ++i){
                if(z.at(i,0) > maxElem){
                    maxElem = z.at(i,0);
                }
            }

            //then we build the result matrix with the stable softmax formula
            double sum = 0.0;
            for(unsigned int i = 0; i < z.getRows(); ++i){
                result.at(i,0) = exp(z.at(i,0) - maxElem); // for numerical stabilit, if we dont substract maxElem we can get overflow
                sum += result.at(i,0);
            }

            //finally we end the calculation by dividing each element by the sum and updating the result matrix
            for(unsigned int i = 0; i < z.getRows(); ++i){
                result.at(i,0) /= sum;
            }

            return result;
        }

    
    public:

        const vector<Matrix>& getActivations() const {
            return activations;
        }

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
                }else if (activation_func == "sign") {
                    activation = activation.map(sign);
                } else if (activation_func == "softmax" && i == weights.size() -1) { //i == weights.size() -1 for added security, we just want to apply softmax in the output layer
                    activation = softmax(activation);
                }else {
                    cerr << "Unknown activation function: " << activation_func << endl;
                    exit(-1);
                }

                activations.push_back(activation);
            }
            return activation;
        }
        
        // MSE, this is use to know how well the NN is performing
        double costFunction(const Matrix& predicted, const Matrix& target) {
            if (predicted.getRows() != target.getRows() || predicted.getCols() != target.getCols()) {
                throw invalid_argument("Matrix dimensions do not match for cost function.");
            }

            double sum = 0.0;
            for (unsigned int i = 0; i < predicted.getRows(); ++i) {
                for (unsigned int j = 0; j < predicted.getCols(); ++j) {
                    double diff = predicted.at(i, j) - target.at(i, j);
                    sum += diff * diff;
                }
            }
            return sum / 2.0;
        }

        //backpropagation function needs to be implemented

        void backpropagate(const Matrix& target, double learning_rate, string activation_func) {
            
            // 1. calculate the output error (delta)
            Matrix output = activations.back();
            Matrix error = (output - target).ScalarMul(2.0); //MSE derivative: 2 * (Output - Target) 
            Matrix gradient = error;
            

            if (activation_func == "sigmoid") {
                Matrix derivative = output.map(dsigmoid);
                gradient = gradient.Hadamard(derivative);
            } 
            else if (activation_func == "tanh") {
                Matrix derivative = output.map(dtanh);
                gradient = gradient.Hadamard(derivative);
            }

            // 2. Backpropagate the error through the network
            for (int i = weights.size() - 1; i >= 0; --i) {

                //calculate gradients for weights and biases
                //prev activation is activations[i]
                Matrix prev_activation = activations[i];
                
                // Delta Weights = Gradient * (prev_activation)^T
                Matrix delta_weights = gradient * prev_activation.transpose();
                
                //delta biases is just the gradient
                Matrix delta_biases = gradient;

                //only calculate the error if we are not in the input layer
                if (i > 0) {
                    // Backpropagate the error: Weights^T * Current_Gradient
                    Matrix weight_transposed = weights[i].transpose();
                    Matrix prev_error = weight_transposed * gradient;
                    
                    // Calculate the derivative of the previous layer
                    Matrix prev_derivative;
                    if (activation_func == "sigmoid") {
                        prev_derivative = activations[i].map(dsigmoid);
                    } else if (activation_func == "tanh") {
                        prev_derivative = activations[i].map(dtanh);
                    } else {
                        
                        // Si es otra, asumimos derivada 1 (identidad) o manejo de errores
                        prev_derivative = activations[i].map([](double x){ return 1.0; }); 
                    }

                    // new gradient = prev_error Hadamard prev_derivative
                    gradient = prev_error.Hadamard(prev_derivative);
                }

                // 3. Update weights and biases
                weights[i] = weights[i] - delta_weights.ScalarMul(learning_rate);
                biases[i] = biases[i] - delta_biases.ScalarMul(learning_rate);
            }
        }
        

        //THIS NEEDS A LOT OF WORK 
        void train(const Matrix& input, const Matrix& target, double learning_rate, string activation_func = "sigmoid") {
            feedforward(input, activation_func);
            backpropagate(target, learning_rate, activation_func);
        }
};

#endif