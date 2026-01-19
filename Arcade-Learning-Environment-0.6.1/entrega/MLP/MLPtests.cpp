#include <cmath>
#include <iostream>
#include "NeuralNetwork.h"
#include <vector>
using namespace std;


class DonutProblem {
    private:
        vector<Matrix> TrainInputs;
        vector<Matrix> TrainTargets;
        vector<Matrix> TestInputs;
        vector<Matrix> TestTargets;
        vector<Matrix> valInputs;
        vector<Matrix> valTargets;

    public:
        DonutProblem(int samples) {
            
            // Definimos qué es el donut
            double innerRadius = 0.5;
            double outerRadius = 0.8;

            // --- GENERAR DATOS DE ENTRENAMIENTO ---
            for (int i = 0; i < samples; ++i) {
                double x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double r = sqrt(x*x + y*y);

                Matrix input(2, 1);
                input.at(0, 0) = x;
                input.at(1, 0) = y;
                TrainInputs.push_back(input);

                Matrix target(2, 1); 
                
                if (r >= innerRadius && r <= outerRadius) {
                    // ES DONUT (Clase 1)
                    target.at(0, 0) = -1.0; // Neurona NO-Donut apagada
                    target.at(1, 0) = 1.0;  // Neurona SI-Donut encendida
                } 
                else {
                    // ES FONDO (Clase 0)
                    target.at(0, 0) = 1.0;  // Neurona NO-Donut encendida
                    target.at(1, 0) = -1.0; // Neurona SI-Donut apagada
                }

                TrainTargets.push_back(target);
            }

            //generate validation data (20% of training samples)

            for (int i = 0; i < samples / 5; ++i) {
                double x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double r = sqrt(x*x + y*y);

                Matrix input(2, 1);
                input.at(0, 0) = x;
                input.at(1, 0) = y;
                valInputs.push_back(input);

                Matrix target(2, 1); 
                
                if (r >= innerRadius && r <= outerRadius) {
                    target.at(0, 0) = -1.0; 
                    target.at(1, 0) = 1.0; 
                } 
                else {
                    target.at(0, 0) = 1.0;
                    target.at(1, 0) = -1.0;
                }

                valTargets.push_back(target);
            }

            // --- GENERAR DATOS DE TEST ---
            for (int i = 0; i < samples; ++i) {
                double x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double r = sqrt(x*x + y*y);

                Matrix input(2, 1);
                input.at(0, 0) = x;
                input.at(1, 0) = y;
                TestInputs.push_back(input);

                Matrix target(2, 1);
                
                if (r >= innerRadius && r <= outerRadius) {
                    target.at(0, 0) = -1.0; 
                    target.at(1, 0) = 1.0; 
                } 
                else {
                    target.at(0, 0) = 1.0;
                    target.at(1, 0) = -1.0;
                }

                TestTargets.push_back(target);
            }
        }
        
        NeuralNetwork createAndTrainNetwork(int epochs = 200, double learning_rate = 0.1,int patience = 15, string activation_func = "tanh") {
            vector<unsigned int> topology = {2, 24, 12, 6, 2}; 
            NeuralNetwork nn(topology);
            cout << "Entrenando la red para el problema del donut..." << endl;
            nn.train(TrainInputs, TrainTargets,valInputs, valTargets, epochs, learning_rate, activation_func, patience, true);
            return nn;
        }

        const vector<Matrix>& getTestInputs() const { return TestInputs; }
        const vector<Matrix>& getTestTargets() const { return TestTargets; }
        const vector<Matrix>& getValInputs() const { return valInputs; }
        const vector<Matrix>& getValTargets() const { return valTargets; }
};

class QuadrantProblem{
    private:
        vector<Matrix> inputs;
        vector<Matrix> targets;
        vector<Matrix> testInputs;
        vector<Matrix> testTargets;
        vector<Matrix> valInputs;
        vector<Matrix> valTargets;

    public:
        QuadrantProblem(int samples) {

            for(int i = 0; i < samples; i++){
                double x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

                Matrix input_matrix(2, 1);
                input_matrix.at(0, 0) = x;
                input_matrix.at(1, 0) = y;

                Matrix target_matrix(4, 1);
                for(int k = 0; k < 4; ++k) {
                    target_matrix.at(k, 0) = 0.0;
                }

                int val = 0;
                if(x >= 0 && y >= 0) val = 0;        // Primer cuadrante
                else if(x < 0 && y >= 0) val = 1;   // Segundo cuadrante
                else if(x < 0 && y < 0) val = 2;    // Tercer cuadrante
                else if(x >= 0 && y < 0) val = 3;   // Cuarto cuadrante

                target_matrix.at(val, 0) = 1.0;

                inputs.push_back(input_matrix);
                targets.push_back(target_matrix);
            }

            //generate validation data (20% of training samples)
            for (int i = 0; i < samples / 5; ++i) {
                double x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

                Matrix input_matrix(2, 1);
                input_matrix.at(0, 0) = x;
                input_matrix.at(1, 0) = y;

                Matrix target_matrix(4, 1);
                for(int k = 0; k < 4; ++k) {
                    target_matrix.at(k, 0) = 0.0;
                }

                int val = 0;
                if(x >= 0 && y >= 0) val = 0;        // Primer cuadrante
                else if(x < 0 && y >= 0) val = 1;   // Segundo cuadrante
                else if(x < 0 && y < 0) val = 2;    // Tercer cuadrante
                else if(x >= 0 && y < 0) val = 3;   // Cuarto cuadrante

                target_matrix.at(val, 0) = 1.0;

                valInputs.push_back(input_matrix);
                valTargets.push_back(target_matrix);
            }

            // Generar datos de prueba
            for(int i = 0; i < samples; i++){
                double x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

                Matrix input_matrix(2, 1);
                input_matrix.at(0, 0) = x;
                input_matrix.at(1, 0) = y;

                Matrix target_matrix(4, 1);
                for(int k = 0; k < 4; ++k) {
                    target_matrix.at(k, 0) = 0.0;
                }

                int val = 0;
                if(x >= 0 && y >= 0) val = 0;        // Primer cuadrante
                else if(x < 0 && y >= 0) val = 1;   // Segundo cuadrante
                else if(x < 0 && y < 0) val = 2;    // Tercer cuadrante
                else if(x >= 0 && y < 0) val = 3;   // Cuarto cuadrante

                target_matrix.at(val, 0) = 1.0;

                testInputs.push_back(input_matrix);
                testTargets.push_back(target_matrix);
            }
        }

        NeuralNetwork createAndTrainNetwork(int epochs = 200, double learning_rate = 0.1, int patience = 15, string activation_func = "tanh") {
            vector<unsigned int> topology = {2, 10, 10, 4}; 
            NeuralNetwork nn(topology);
            cout << "Entrenando la red para el problema del cuadrante..." << endl;
            nn.train(inputs, targets, valInputs, valTargets, epochs, learning_rate, activation_func, patience, true);
            return nn;
        }

        const vector<Matrix>& getTestInputs() const { return testInputs; }
        const vector<Matrix>& getTestTargets() const { return testTargets; }
        const vector<Matrix>& getValInputs() const { return valInputs; }
        const vector<Matrix>& getValTargets() const { return valTargets; }
};

class ParityProblem {
    private:

        vector<Matrix> TrainInputs;
        vector<Matrix> TrainTargets;
        vector<Matrix> TestInputs;
        vector<Matrix> TestTargets;
        vector<Matrix> valInputs;
        vector<Matrix> valTargets;
        int nBits; // Guardamos el número de bits para la topología
        

    public:
        ParityProblem(int n, int samples) : nBits(n) {
            
            // --- GENERAR DATOS DE ENTRENAMIENTO ---
            for (int i = 0; i < samples; ++i) {
                
                Matrix input(nBits, 1);
                int onesCount = 0;

                for (int b = 0; b < nBits; ++b) {
                    int bit = rand() % 2;
                    
                    if (bit == 0) input.at(b, 0) = -1.0;
                    else {
                        input.at(b, 0) = 1.0;
                        onesCount++;
                    }
                }
                TrainInputs.push_back(input);

                Matrix target(1, 1);

                if (onesCount % 2 != 0) {
                    target.at(0, 0) = 1.0; 
                } else {
                    target.at(0, 0) = -1.0;
                }

                TrainTargets.push_back(target);
            }

            // --- GENERAR DATOS DE VALIDACION ---
            for (int i = 0; i < samples / 5; ++i){
                Matrix input(nBits, 1);
                int onesCount = 0;

                for (int b = 0; b < nBits; ++b) {
                    int bit = rand() % 2; 
                    if (bit == 0) input.at(b, 0) = -1.0;
                    else {
                        input.at(b, 0) = 1.0;
                        onesCount++;
                    }
                }
                valInputs.push_back(input);

                Matrix target(1, 1);
                
                if (onesCount % 2 != 0) {
                    target.at(0, 0) = 1.0; 
                } else {
                    target.at(0, 0) = -1.0;
                }

                valTargets.push_back(target);
            }

            // --- GENERAR DATOS DE TEST ---
            for (int i = 0; i < samples; ++i) {
                
                Matrix input(nBits, 1);
                int onesCount = 0;

                for (int b = 0; b < nBits; ++b) {
                    int bit = rand() % 2; 
                    if (bit == 0) input.at(b, 0) = -1.0;
                    else {
                        input.at(b, 0) = 1.0;
                        onesCount++;
                    }
                }
                TestInputs.push_back(input);

                Matrix target(1, 1);
                
                if (onesCount % 2 != 0) {
                    target.at(0, 0) = 1.0; 
                } else {
                    target.at(0, 0) = -1.0;
                }

                TestTargets.push_back(target);
            }
        }

        NeuralNetwork createAndTrainNetwork(int epochs = 1000, double learning_rate = 0.01, int patience = 15, string activation_func = "tanh") {

            vector<unsigned int> topology = {(unsigned int)nBits, 24, 24, 1}; 
            
            NeuralNetwork nn(topology);
            
            cout << "Entrenando la red para Paridad de " << nBits << " bits..." << endl;
            nn.train(TrainInputs, TrainTargets, valInputs, valTargets, epochs, learning_rate, activation_func, patience, true);
            
            return nn;
        }

        const vector<Matrix>& getTestInputs() const { return TestInputs; }
        const vector<Matrix>& getTestTargets() const { return TestTargets; }
        const vector<Matrix>& getValInputs() const { return valInputs; }
        const vector<Matrix>& getValTargets() const { return valTargets; }
};

int main(int argc, char* argv[]) {
    
    bool donut = (argc == 3 && string(argv[1]) == "donut");
    bool quadrant = (argc == 3 && string(argv[1]) == "quadrant");
    bool parity = (argc == 4 && string(argv[1]) == "parity");

     // 1. SELECCIONAR EL PROBLEMA A RESOLVER
   
    if (!donut && !quadrant && !parity) {
        cout << "Uso: " << argv[0] << " [donut|quadrant|parity <nBits>] <samples>" << endl;
        return -1;
    }

    int correct = 0;
    int inputSize = 0;

    if(donut) {
        cout << "========= PRUEBA DE APRENDIZAJE - DONUT =========" << endl;
        DonutProblem problem(stoi(argv[2]));
        
        // 2. CREAR Y ENTRENAR LA RED EN BASE A LOS DATOS GENERADOS DEl PROBLEMA SELECCIONADO
        NeuralNetwork nn = problem.createAndTrainNetwork(120, 0.01, 15, "tanh");
        

        // 3. VERIFICAR RESULTADOS EN DATOS DE TEST
        const vector<Matrix>& testInputs = problem.getTestInputs();
        const vector<Matrix>& testTargets = problem.getTestTargets();
        correct = 0;
        inputSize = testInputs.size();

        for(size_t i = 0; i < testInputs.size(); ++i) {
            
            // ESTE CODIGO ES PARA PROBLEMAS DE CLASIFICACION MULTINOMIAL
            //realiza la prediccion
            int prediction = nn.predictOne(testInputs[i], "tanh");  
    
            int realClass = 0;
            double maxTargetVal = testTargets[i].at(0, 0);

            for(int k = 1; k < testTargets[i].getRows(); ++k) {
                if(testTargets[i].at(k, 0) > maxTargetVal) { 
                    maxTargetVal = testTargets[i].at(k, 0);
                    realClass = k;
                }
            }

            if (prediction == realClass) {
                correct++;
            }
        }

    } else if (quadrant) {
        cout << "========= PRUEBA DE APRENDIZAJE - CUADRANTES =========" << endl;
        QuadrantProblem problem(stoi(argv[2]));

        
        // 2. CREAR Y ENTRENAR LA RED EN BASE A LOS DATOS GENERADOS DEl PROBLEMA SELECCIONADO
        NeuralNetwork nn = problem.createAndTrainNetwork(100, 0.01,15, "tanh");

        // 3. VERIFICAR RESULTADOS EN DATOS DE TEST
        const vector<Matrix>& testInputs = problem.getTestInputs();
        const vector<Matrix>& testTargets = problem.getTestTargets();
        correct = 0;
        inputSize = testInputs.size();

        for(size_t i = 0; i < testInputs.size(); ++i) {
            
            // ESTE CODIGO ES PARA PROBLEMAS DE CLASIFICACION MULTINOMIAL
            //realiza la prediccion
            int prediction = nn.predictOne(testInputs[i], "tanh");  
    
            int realClass = 0;
            double maxTargetVal = testTargets[i].at(0, 0);

            for(int k = 1; k < testTargets[i].getRows(); ++k) {
                if(testTargets[i].at(k, 0) > maxTargetVal) { 
                    maxTargetVal = testTargets[i].at(k, 0);
                    realClass = k;
                }
            }

            if (prediction == realClass) {
                correct++;
            }
        }

    } else if (parity) {
        string auxString = "";
        if (stoi(argv[2]) == 2) auxString = "(XOR) ";

        cout << "========= PRUEBA DE APRENDIZAJE - PARIDAD DE " << argv[2] << " BITS " << auxString << "=========" << endl;
        ParityProblem problem(stoi(argv[2]), stoi(argv[3]));

            
        // 2. CREAR Y ENTRENAR LA RED EN BASE A LOS DATOS GENERADOS DEl PROBLEMA SELECCIONADO
        NeuralNetwork nn = problem.createAndTrainNetwork(100, 0.01,15, "tanh");

        // 3. VERIFICAR RESULTADOS EN DATOS DE TEST
        const vector<Matrix>& testInputs = problem.getTestInputs();
        const vector<Matrix>& testTargets = problem.getTestTargets();
        correct = 0;
        inputSize = testInputs.size();

        for(size_t i = 0; i < testInputs.size(); ++i) {

            //ESTE CODIGO ES PARA EL PROBLEMA DE PARIDAD (SALIDA UNICA)
            Matrix output = nn.feedforward(testInputs[i], "tanh");
            double predictedVal = output.at(0, 0);
            double targetVal = testTargets[i].at(0, 0);

            // Acierto si tienen el mismo signo
            // (Ambos positivos O Ambos negativos)
            if ((predictedVal > 0 && targetVal > 0) || (predictedVal < 0 && targetVal < 0)) {
                correct++;
            }
        }
    }else {
        cout << "Problema no reconocido." << endl;
        return -1;
    }

    double accuracy = (double)correct / inputSize * 100.0;
    cout << "\nPrecision total en datos de test: " << accuracy << "%" << endl;

    HoeffdingInequality(inputSize, 1 - (accuracy / 100.0), 0.05);

    return 0;
}


