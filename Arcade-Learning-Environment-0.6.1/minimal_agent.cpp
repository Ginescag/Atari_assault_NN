#include <iostream>
#include <cmath>
#include <cstdint>
#include <map>
#include <fstream>
#include <string>
#include <filesystem>
#include "src/ale_interface.hpp"
using namespace std;
// Constants
constexpr uint32_t maxSteps = 500;

///////////////////////////////////////////////////////////////////////////////
/// Get info from RAM
///////////////////////////////////////////////////////////////////////////////
int32_t getPlayerX(ALEInterface& alei) {
   return alei.getRAM().get(72) + ((rand() % 3) - 1);
}

int32_t getBallX(ALEInterface& alei) {
   return alei.getRAM().get(99) + ((rand() % 3) - 1);
}

///////////////////////////////////////////////////////////////////////////////
/// Do Next Agent Step
///////////////////////////////////////////////////////////////////////////////
reward_t agentStep(ALEInterface& alei) {
   static constexpr int32_t wide { 9 };
   static int32_t lives { alei.lives() };
   reward_t reward{0};

   // When we loose a live, we need to press FIRE to start again
   if (alei.lives() < lives) {
      lives = alei.lives();
      alei.act(PLAYER_A_FIRE);
   }

   // Apply rules.
   auto playerX { getPlayerX(alei) };
   auto ballX   { getBallX(alei)   };
   
   if       (ballX < playerX + wide) { reward = alei.act(PLAYER_A_LEFT);   }
   else if  (ballX > playerX + wide) { reward = alei.act(PLAYER_A_RIGHT);  }
   
   return reward + alei.act(PLAYER_A_NOOP);
}

///////////////////////////////////////////////////////////////////////////////
/// Print usage and exit
/// Definimos las posibilidades de uso
/// En modo heatmap guarda en el fichero destino las variaciones en la RAM
/// En modo dataset se juega en modo manual y se guardan los datos en el fichero destino
/// En modo train se realiza el entrenamiento a partir de los datos del fichero de origen y guarda los pesos de la red en el fichero destino  
///////////////////////////////////////////////////////////////////////////////

void usage(char const* pname) {
   std::cerr
      << "\nUSAGE:\n" 
      << "   " << pname << " <romfile>" << " (heatmap | dataset | train <ORGfile>) <DSTfile>\n";
   exit(-1);
}


//PARA USAR ESTA FUNCION TIENES QUE INICIALIZAR PREVRAM
void getRAMFreq(map<int, int>& RAMmap, ALEInterface& alei, auto& prevRAM){
   auto aux = alei.getRAM();
   for(int i = 0; i < 128; i++){
      if(aux.get(i) != prevRAM.get(i)){
         RAMmap[i] = RAMmap[i] + 1;
      }
   }
   prevRAM = aux;
}

//para pasar del diccionario a un fichero para hacer el heatmap
void mapToFile(string& filename, map<int, int>& RAMmap){
   ofstream RAMfile(filename);
   for(int i = 0; i < 128; i++){
      RAMfile << RAMmap[i] << "\n";
   }
   RAMfile.close();
}

///////////////////////////////////////////////////////////////////////////////
/// MAIN PROGRAM
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
   reward_t totalReward{};
   ALEInterface alei{};

   bool trainMode = (argc == 5 && string(argv[2]) == "train");
   bool datasetMode = (argc == 4 && string(argv[2]) == "dataset");
   bool heatmapMode = (argc == 4 && string(argv[2]) == "heatmap");
   
   // Check parameters and modes
   if (!trainMode && !datasetMode && !heatmapMode)
      usage(argv[0]);

   // Configure alei object.
   alei.setInt  ("random_seed", 0);
   alei.setFloat("repeat_action_probability", 0);
   alei.setBool ("display_screen", true);
   alei.setBool ("sound", true);
   alei.loadROM (argv[1]);
  
  
   if(heatmapMode){
   string RAMfile = string(argv[3])
   auto prevRAM = alei.getRAM();
   map <int, int> RAMmap;
   string fileName = RAMfile;

   // Init
   std::srand(static_cast<uint32_t>(std::time(0)));

   // Main loop
      alei.act(PLAYER_A_FIRE);
      uint32_t step{};
      while ( !alei.game_over() && step < maxSteps ) { 
         totalReward += agentStep(alei);
         getRAMFreq(RAMmap, alei, prevRAM);
         ++step;
      }

      std::cout << "Steps: " << step << std::endl;
      std::cout << "Reward: " << totalReward << std::endl;

      mapToFile(fileName, RAMmap);
   }


   return 0;
}
