#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
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
                cerr << "Matrix dimensions do not match for multiplication." << endl;
                exit(-1);
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
                cerr << "Matrix dimensions do not match for addition." << endl;
                exit(-1);
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
                cerr << "Matrix dimensions do not match for subtraction." << endl;
                exit(-1);
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
                cerr << "Matrix dimensions do not match for Hadamard product." << endl;
                exit(-1);
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

        void XavierInit(){}

        void RandIntMat(int minVal, int maxVal){
            for (unsigned int i = 0; i < rows; ++i) {
                for (unsigned int j = 0; j < cols; ++j) {
                    data[i][j] = rand() % (maxVal - minVal + 1) + minVal; // Random integers between minVal and maxVal
                }
            }
        }

        void xavierInit(){}

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
class DataHelper{
    private:
        string filename;
        vector<Matrix> inputs;
        vector<Matrix> targets;

    public:
        DataHelper(const string& file) : filename(file) {
            srand(time(0));

            ifstream infile(filename);
            if (infile.is_open()) {
                string line;
                
                while (getline(infile, line)) {
                    if (line.empty()) continue;
                    
                    stringstream ss(line);
                    
                    Matrix input_matrix(128, 1); 
                    Matrix target_matrix(6, 1); 

                    int val;
                    
                    for (int i = 0; i < 128; ++i) {
                        ss >> val;
                        input_matrix.at(i, 0) = (static_cast<double>(val) - 128.0) / 128.0; //we normalize the input between -1 and 1
                    }
                    
                    if (!(ss >> val)) continue; 
                    
                    int repeticiones = 1;

                    if (val == 0) { 
                        if ((rand() % 100) < 80) repeticiones = 0; 
                    }
                    else if (val == 1 || val == 2) { 
                        repeticiones = 15; 
                    }
                    else if (val == 3) {
                        repeticiones = 2; 
                    }

                    if (repeticiones > 0 && val >= 0 && val < 6) {
                        target_matrix.at(val, 0) = 1.0;

                        for(int k = 0; k < repeticiones; ++k) {
                            inputs.push_back(input_matrix);
                            targets.push_back(target_matrix);
                        }
                    }
                }

                infile.close();
            }
            else {
                cerr << "ERROR: Could not open data file: " << filename << endl;
                exit(-1);
            }
        }

        const vector<Matrix>& getInputs() const {
            return inputs;
        }

        const vector<Matrix>& getTargets() const {
            return targets;
        }

        int getOutputLayerSize() const {
            return (targets.empty()) ? 0 : targets[0].getRows(); //number of rows in target matrix
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
        

        //WE WONT REALLY USE THIS ACTIVATION FUNCTION BUT JUST IN CASE
        //------------------------------------------------------------------------------------
        //------------------------------------------------------------------------------------
        static double sign(double x) {
            return (x >= 0) ? 1.0 : -1.0;
        }

        static double dsign(double y) {
            return 0.0; // Derivative is zero almost everywhere
        }
        //------------------------------------------------------------------------------------ 
        //------------------------------------------------------------------------------------
    
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

        NeuralNetwork() {
            //default constructor
            layers = {};
            weights = {};
            biases = {};
            activations = {};
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

        NeuralNetwork(const string& filename) {
            if (!loadModel(filename)) {
                cerr << "Error: Could not load model from " << filename << ". Empty model initialized." << endl;
                layers = {};
                weights = {};
                biases = {};
                activations = {};
            }
        }


        Matrix feedforward(const Matrix& input, string activation_func = "tanh") {
            activations.clear();
            activations.push_back(input);
            Matrix activation = input;

            for(int i = 0; i < weights.size(); ++i) {
                activation = (weights[i] * activation) + biases[i];
                if (activation_func == "sigmoid") {
                    activation = activation.map(sigmoid);
                }else {
                    activation = activation.map(tanh); //default to tanh
                }
                activations.push_back(activation);
            }
            return activation;
        }
        
        // MSE, this is use to know how well the NN is performing
        double costFunction(const Matrix& predicted, const Matrix& target) {
            if (predicted.getRows() != target.getRows() || predicted.getCols() != target.getCols()) {
                cerr << "Matrix dimensions do not match for cost function." << endl;
                exit(-1);
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

        void backpropagate(const Matrix& target, double learning_rate, string activation_func = "tanh") {
            
            // 1. calculate the output error (delta)
            Matrix output = activations.back();
            Matrix error = (output - target).ScalarMul(2.0); //MSE derivative: 2 * (Output - Target) 
            Matrix gradient = error;


            if (activation_func == "sigmoid") {
                Matrix derivative = output.map(dsigmoid);
                gradient = gradient.Hadamard(derivative);
            } 
            else {
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
                    } 
                    else {
                        prev_derivative = activations[i].map(dtanh);
                    }

                    // new gradient = prev_error Hadamard prev_derivative
                    gradient = prev_error.Hadamard(prev_derivative);
                }

                // 3. Update weights and biases
                weights[i] = weights[i] - delta_weights.ScalarMul(learning_rate);
                biases[i] = biases[i] - delta_biases.ScalarMul(learning_rate);
            }
        }
        
        void train(const vector<Matrix>& inputs, const vector<Matrix>& targets, int epochs, double learning_rate, string activation_func = "tanh", bool verbose = true) {
            // Basic validation
            if (inputs.size() != targets.size()) {
                cerr << "Error: Number of inputs and targets do not match." << endl;
                return;
            }

            for (int epoch = 1; epoch <= epochs; ++epoch) {
                double total_loss = 0.0;

                for (size_t i = 0; i < inputs.size(); ++i) {

                    Matrix output = feedforward(inputs[i], activation_func);
                    
                    total_loss += costFunction(output, targets[i]);

                    backpropagate(targets[i], learning_rate, activation_func);
                }

                if (verbose &&(epoch % 10 == 0 || epoch == 1 || epoch == epochs)) {
                     cout << "Epoch: " << epoch  << "/" << epochs 
                          << " | Avg Error (Loss): " << total_loss / inputs.size() << endl;
                }
            }
            cout << "Training completed." << endl;
        }

        vector<int> predict(const vector<Matrix>& inputs, string activation_func = "tanh") {
            vector<int> predictions;
            
            for (const auto& input : inputs) {
                // 1. Obtener salida de la red
                Matrix output = feedforward(input, activation_func);
                
                // 2. ArgMax: Buscar el índice con el valor más alto
                int maxIndex = 0;
                double maxValue = output.at(0, 0);
                
                for (int i = 1; i < output.getRows(); ++i) {
                    if (output.at(i, 0) > maxValue) {
                        maxValue = output.at(i, 0);
                        maxIndex = i;
                    }
                }
                
                predictions.push_back(maxIndex);
            }
            return predictions;
        }

        int predictOne(const Matrix& input, string activation_func = "tanh") {
            // 1. Obtener salida de la red
            Matrix output = feedforward(input, activation_func);
            
            // 2. ArgMax: Buscar el índice con el valor más alto
            int maxIndex = 0;
            double maxValue = output.at(0, 0);
            
            for (int i = 1; i < output.getRows(); ++i) {
                if (output.at(i, 0) > maxValue) {
                    maxValue = output.at(i, 0);
                    maxIndex = i;
                }
            }
            
            return maxIndex;
        }

        
        bool saveModel(const string& filename) const {
            ofstream file(filename);
            if (!file.is_open()) return false;

            //save topology
            file << layers.size() << endl;
            for (unsigned int size : layers) {
                file << size << " ";
            }
            file << endl;

            // 2. Guardar Pesos
            // Recorremos cada matriz de pesos
            for (const auto& w : weights) {
                file << w.getRows() << " " << w.getCols() << endl; // Cabecera de matriz
                for (unsigned int i = 0; i < w.getRows(); ++i) {
                    for (unsigned int j = 0; j < w.getCols(); ++j) {
                        file << w.at(i, j) << " ";
                    }
                    file << endl;
                }
            }

            // 3. Guardar Sesgos (Biases)
            for (const auto& b : biases) {
                file << b.getRows() << " " << b.getCols() << endl;
                for (unsigned int i = 0; i < b.getRows(); ++i) {
                    for (unsigned int j = 0; j < b.getCols(); ++j) {
                        file << b.at(i, j) << " ";
                    }
                    file << endl;
                }
            }

            file.close();
            cout << "Model saved successfully to: " << filename << endl;
            return true;
        }

        bool loadModel(const string& filename) {
            ifstream file(filename);
            if (!file.is_open()) return false;

            // 1. Cargar Topología
            unsigned int numLayers;
            file >> numLayers;
            
            layers.clear();
            for (unsigned int i = 0; i < numLayers; ++i) {
                unsigned int size;
                file >> size;
                layers.push_back(size);
            }

            // 2. Reconstruir estructura (vaciar y redimensionar vectores)
            weights.clear();
            biases.clear();
            activations.clear(); // Limpiar activaciones antiguas

            // 3. Cargar Pesos
            // Sabemos que hay (numLayers - 1) matrices de pesos
            for (unsigned int i = 0; i < numLayers - 1; ++i) {
                unsigned int rows, cols;
                file >> rows >> cols;
                
                Matrix w(rows, cols);
                for (unsigned int r = 0; r < rows; ++r) {
                    for (unsigned int c = 0; c < cols; ++c) {
                        file >> w.at(r, c);
                    }
                }
                weights.push_back(w);
            }

            // 4. Cargar Sesgos
            for (unsigned int i = 0; i < numLayers - 1; ++i) {
                unsigned int rows, cols;
                file >> rows >> cols;
                
                Matrix b(rows, cols);
                for (unsigned int r = 0; r < rows; ++r) {
                    for (unsigned int c = 0; c < cols; ++c) {
                        file >> b.at(r, c);
                    }
                }
                biases.push_back(b);
            }

            file.close();
            cout << "Model loaded successfully from: " << filename << endl;
            return true;
        }

};

#endif