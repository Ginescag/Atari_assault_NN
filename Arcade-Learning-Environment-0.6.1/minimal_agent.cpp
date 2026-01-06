#include <iostream>
#include <cmath>
#include <cstdint>
#include <map>
#include <fstream>
#include <string>
#include <filesystem>
#include "src/ale_interface.hpp"
#include <SDL/SDL.h>
using namespace std;
// Constants
constexpr uint32_t maxSteps = 8000000;


string getPlayerAction(ALEInterface& alei){
   string action = "PLAYER_A_NOOP";
   Uint8* keystates = SDL_GetKeyState(NULL);


   if (keystates[SDLK_SPACE] && keystates[SDLK_RIGHT]){
      action = "PLAYER_A_RIGHTFIRE";
   }
   else if (keystates[SDLK_SPACE] && keystates[SDLK_LEFT]){
      action = "PLAYER_A_LEFTFIRE";
   }
   else if (keystates[SDLK_UP]){
      action = "PLAYER_A_UPFIRE";
   }
   else if (keystates[SDLK_LEFT]){
      action = "PLAYER_A_LEFT";
   } 
   else if (keystates[SDLK_RIGHT]){
      action = "PLAYER_A_RIGHT";
   }

   return action;
}

reward_t manualStep(ALEInterface& alei){
   Action action = PLAYER_A_NOOP;
   Uint8* keystates = SDL_GetKeyState(NULL);


   if (keystates[SDLK_SPACE] && keystates[SDLK_RIGHT]){
      action = PLAYER_A_RIGHTFIRE;
   }
   else if (keystates[SDLK_SPACE] && keystates[SDLK_LEFT]){
      action = PLAYER_A_LEFTFIRE;
   }
   else if (keystates[SDLK_UP]){
      action = PLAYER_A_UPFIRE;
   }
   else if (keystates[SDLK_LEFT]){
      action = PLAYER_A_LEFT;
   } 
   else if (keystates[SDLK_RIGHT]){
      action = PLAYER_A_RIGHT;
   }
   else if (keystates[SDLK_ESCAPE]){
      cout << "Exiting game" << endl;
      exit(0);
   }

   return alei.act(action);
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

//this retrieves every RAM byte at a given step and the action the player made in that state and saves that info inside the file 
void collectData(ALEInterface& alei, string filename){
   auto RAMArr = alei.getRAM();
   string action = getPlayerAction(alei);
   ofstream DataFile(filename, ios::app);
   
   if(DataFile.is_open()){

   for(int i = 0; i < 128; i++){
      DataFile << static_cast<int>(RAMArr.get(i)) << " ";
   }

   DataFile << action << "\n";
   DataFile.close();

   }else{
      cerr << "ERROR: could not open Data File";
   }

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
      string RAMfile = string(argv[3]);
      auto prevRAM = alei.getRAM();
      map <int, int> RAMmap;

      // Init
      std::srand(static_cast<uint32_t>(std::time(0)));
      uint32_t step{};
      bool manual = {true};
      SDL_Event ev;
      int32_t lives {alei.lives() };

      while ( !alei.game_over() && step < maxSteps ) { 
         while(SDL_PollEvent(&ev)){
            if(ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_m){
               manual = !manual;
            }
         }

         if(alei.lives() < lives){
            lives = alei.lives();
            alei.act(PLAYER_A_FIRE);
         }

         if(manual){
            totalReward += manualStep(alei);
         }else{
            totalReward += 1;
         }

         getRAMFreq(RAMmap, alei, prevRAM);
         ++step;
      }

      std::cout << "RAM FREQ COLLECTED. RUN HEATMAP SCRIPT" << step << std::endl;
      mapToFile(RAMfile, RAMmap);
   }

   if(datasetMode){
      string DataFile = string(argv[3]);

      // Init
      std::srand(static_cast<uint32_t>(std::time(0)));


      uint32_t step{};
      bool manual = {true};
      SDL_Event ev;
      int32_t lives {alei.lives() };

      while ( !alei.game_over() && step < maxSteps ) { 
         while(SDL_PollEvent(&ev)){
            if(ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_m){
               manual = !manual;
            }
         }

         if(alei.lives() < lives){
            lives = alei.lives();
            alei.act(PLAYER_A_FIRE);
         }

         if(manual){
            totalReward += manualStep(alei);
         }else{
            totalReward += 1;
         }

         collectData(alei, DataFile);
         ++step;
      }

      std::cout << "DATA COLLECTED" << std::endl;

   }

   if(trainMode){
      //retrieves info from the data set, normalizes it (y label), trains the model on top of a MLP library (TO-DO)
   }

   return 0;
}
