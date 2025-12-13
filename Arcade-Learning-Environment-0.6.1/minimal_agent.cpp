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
///////////////////////////////////////////////////////////////////////////////
void usage(char const* pname) {
   std::cerr
      << "\nUSAGE:\n" 
      << "   " << pname << " <romfile>\n";
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
   std::cout << "Working dir: " << std::filesystem::current_path() << std::endl;
   // Check input parameter
   if (argc != 2)
      usage(argv[0]);

   // Configure alei object.
   alei.setInt  ("random_seed", 0);
   alei.setFloat("repeat_action_probability", 0);
   alei.setBool ("display_screen", true);
   alei.setBool ("sound", true);
   alei.loadROM (argv[1]);

   auto prevRAM = alei.getRAM();
   map <int, int> RAMmap;
   string fileName = "RamFILE.txt";

   // Init
   std::srand(static_cast<uint32_t>(std::time(0)));

   // Main loop
   {
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
