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

                // CORRECCIÓN 1: Matriz de tamaño 2
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

            // --- GENERAR DATOS DE TEST ---
            for (int i = 0; i < samples / 5; ++i) {
                double x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double r = sqrt(x*x + y*y);

                Matrix input(2, 1);
                input.at(0, 0) = x;
                input.at(1, 0) = y;
                TestInputs.push_back(input);

                // CORRECCIÓN 2: También tamaño 2 en el Test
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
        
        // ... (El resto de funciones createAndTrainNetwork y getters están bien) ...
        NeuralNetwork createAndTrainNetwork(int epochs = 200, double learning_rate = 0.1, string activation_func = "tanh") {
            vector<unsigned int> topology = {2, 10, 10, 2}; 
            NeuralNetwork nn(topology);
            cout << "Entrenando la red para el problema del donut..." << endl;
            nn.train(TrainInputs, TrainTargets, epochs, learning_rate, activation_func, false);
            return nn;
        }

        const vector<Matrix>& getTestInputs() const { return TestInputs; }
        const vector<Matrix>& getTestTargets() const { return TestTargets; }
};

int main() {
    cout << "=== PRUEBA DE APRENDIZAJE: PROBLEMA DEL DONUT ===" << endl;

    // 1. PREPARAR DATOS DEL PROBLEMA DEL DONUT
    DonutProblem donut(1000);

    // 2. CREAR Y ENTRENAR LA RED
    NeuralNetwork nn = donut.createAndTrainNetwork(500, 0.01, "tanh");

    // 3. VERIFICAR RESULTADOS EN DATOS DE TEST
    const vector<Matrix>& testInputs = donut.getTestInputs();
    const vector<Matrix>& testTargets = donut.getTestTargets();

    int correct = 0;
    for(size_t i = 0; i < testInputs.size(); ++i) {
        
        // Ahora devuelve 0 (Fondo) o 1 (Donut)
        int prediction = nn.predictOne(testInputs[i], "tanh"); 
        
        // Miramos cuál era la real buscando el 1.0 en el target
        int realClass = (testTargets[i].at(1, 0) > 0.9) ? 1 : 0;

        if (prediction == realClass) {
            correct++;
        }
    }

    double accuracy = (double)correct / testInputs.size() * 100.0;
    cout << "\nPrecision total en datos de test: " << accuracy << "%" << endl;

    return 0;
}


