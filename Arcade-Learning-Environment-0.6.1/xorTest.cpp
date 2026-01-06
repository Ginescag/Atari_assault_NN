#include <iostream>
#include <vector>
#include "NeuralNetwork.h" // Asegúrate de que tu fichero se llama así

using namespace std;

int main() {
    cout << "=== PRUEBA DE APRENDIZAJE: OPERACION XOR ===" << endl;

    // 1. PREPARAR DATOS XOR
    // Inputs: (0,0), (0,1), (1,0), (1,1)
    vector<Matrix> inputs;
    vector<Matrix> targets;

    // Caso 0,0 -> 0
    Matrix i1(2, 1); i1.at(0,0) = 0; i1.at(1,0) = 0;
    Matrix t1(1, 1); t1.at(0,0) = 0;
    inputs.push_back(i1); targets.push_back(t1);

    // Caso 0,1 -> 1
    Matrix i2(2, 1); i2.at(0,0) = 0; i2.at(1,0) = 1;
    Matrix t2(1, 1); t2.at(0,0) = 1;
    inputs.push_back(i2); targets.push_back(t2);

    // Caso 1,0 -> 1
    Matrix i3(2, 1); i3.at(0,0) = 1; i3.at(1,0) = 0;
    Matrix t3(1, 1); t3.at(0,0) = 1;
    inputs.push_back(i3); targets.push_back(t3);

    // Caso 1,1 -> 0
    Matrix i4(2, 1); i4.at(0,0) = 1; i4.at(1,0) = 1;
    Matrix t4(1, 1); t4.at(0,0) = 0;
    inputs.push_back(i4); targets.push_back(t4);


    // 2. CONFIGURAR LA RED
    // Topología: 2 entradas -> 4 ocultas -> 1 salida
    // (Necesitamos capas ocultas para resolver XOR)
    vector<unsigned int> topology = {2, 4, 1};
    NeuralNetwork nn(topology);


    // 3. ENTRENAR
    cout << "\nEntrenando..." << endl;
    // 5000 épocas, learning rate 0.1, función tanh
    nn.train(inputs, targets, 5000, 0.1, "tanh");


    // 4. VERIFICAR RESULTADOS
    cout << "\n=== RESULTADOS FINALES ===" << endl;
    for(size_t i = 0; i < inputs.size(); ++i) {
        Matrix out = nn.feedforward(inputs[i], "tanh");
        
        cout << "Entrada: " << inputs[i].at(0,0) << ", " << inputs[i].at(1,0);
        cout << " | Target: " << targets[i].at(0,0);
        cout << " | Prediccion: " << out.at(0,0) << endl;
    }

    return 0;
}