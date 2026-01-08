#include <iostream>
#include <cmath>
#include <cstdint>
#include <map>
#include <fstream>
#include <string>
#include <filesystem>
#include "src/ale_interface.hpp"
#include <SDL/SDL.h>
#include "NeuralNetwork.h"
using namespace std;
// Constants
constexpr uint32_t maxSteps = 500;


string getPlayerAction(ALEInterface& alei){
   string action = "0"; //"PLAYER_A_NOOP";
   Uint8* keystates = SDL_GetKeyState(NULL);


   if (keystates[SDLK_SPACE] && keystates[SDLK_RIGHT]){
      action = "1"; //"PLAYER_A_RIGHTFIRE";
   }
   else if (keystates[SDLK_SPACE] && keystates[SDLK_LEFT]){
      action = "2"; //"PLAYER_A_LEFTFIRE";
   }
   else if (keystates[SDLK_UP]){
      action = "3"; //"PLAYER_A_UPFIRE";
   }
   else if (keystates[SDLK_LEFT]){
      action = "4"; //"PLAYER_A_LEFT";
   } 
   else if (keystates[SDLK_RIGHT]){
      action = "5"; //"PLAYER_A_RIGHT";
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
/// En modo model se carga el modelo guardado en el fichero de origen y juega en modo automático
///////////////////////////////////////////////////////////////////////////////

void usage(char const* pname) {
   std::cerr
      << "\nUSAGE:\n" 
      << "   " << pname << " <romfile>" << " (heatmap <DSTfile> | dataset <DSTfile> | model <ORGfile> | train <ORGfile> <DSTfile>)\n";
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

//Parses RAM changes to frequency map
void getRAMFreq(map<int, int>& RAMmap, ALEInterface& alei, auto& prevRAM){
   auto aux = alei.getRAM();
   for(int i = 0; i < 128; i++){
      if(aux.get(i) != prevRAM.get(i)){
         RAMmap[i] = RAMmap[i] + 1;
      }
   }
   prevRAM = aux;
}

//parses map to file
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
   bool modelMode = (argc == 4 && string(argv[2]) == "model");

   // Check parameters and modes
   if (!trainMode && !datasetMode && !heatmapMode && !modelMode)
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

      string ORGfile = string (argv[3]);
      string DSTfile = string (argv[4]);

      DataHelper dataHelper(ORGfile);

      NeuralNetwork nn({128, 64, 32, dataHelper.getOutputLayerSize()});

      nn.train(dataHelper.getInputs(), dataHelper.getTargets(),200, 0.01, "tanh");

      nn.saveModel(DSTfile);
   }

   if(modelMode){
      string ORGfile = string (argv[3]);

      NeuralNetwork nn(ORGfile);

      cout << "MODEL LOADED. STARTING AUTOMATIC PLAY" << endl;

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
            //automatic play using the model
            auto RAMArr = alei.getRAM();
            Matrix input_matrix(128, 1);
            for(int i = 0; i < 128; i++){
               input_matrix.at(i, 0) = static_cast<double>(RAMArr.get(i)) / 255.0; //normalize RAM byte value
            }

            int predictedAction = nn.predictOne(input_matrix, "tanh");

               Action action = PLAYER_A_NOOP;

            switch (predictedAction){
               case 0:
                  action = PLAYER_A_NOOP;
                  break;
               case 1:
                  action = PLAYER_A_RIGHTFIRE;
                  break;
               case 2:
                  action = PLAYER_A_LEFTFIRE;
                  break;
               case 3:
                  action = PLAYER_A_UPFIRE;
                  break;
               case 4:
                  action = PLAYER_A_LEFT;
                  break;
               case 5:
                  action = PLAYER_A_RIGHT;
                  break;
               default:
                  action = PLAYER_A_NOOP; //in case of error
                  break;
            }

            totalReward += alei.act(action);
         }
         
         ++step;
      }
   }  

   return 0;
}
