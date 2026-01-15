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
#include "ga_weights.h"

using namespace std;
// Constants
constexpr uint32_t maxSteps = 500;

static constexpr int GA_INPUTS  = 128;
static constexpr int GA_HIDDEN  = 32;
static constexpr int GA_OUTPUTS = 7;

static const Action GA_ACTIONS[GA_OUTPUTS] = {
    PLAYER_A_NOOP,
    PLAYER_A_FIRE,
    PLAYER_A_LEFT,
    PLAYER_A_RIGHT,
    PLAYER_A_LEFTFIRE,
    PLAYER_A_RIGHTFIRE,
    PLAYER_A_UPFIRE
};

static inline double normalize_ram(uint8_t b) {
    return (double(b) / 127.5) - 1.0; // [0,255] -> [-1,1]
}

static int argmax_colvec(const Matrix& out) {
    int best = 0;
    double bestv = out.at(0,0);
    for (unsigned i = 1; i < out.getRows(); ++i) {
        double v = out.at(i,0);
        if (v > bestv) { bestv = v; best = (int)i; }
    }
    return best;
}

static double runGAMode(ALEInterface& ale, const std::string& activation_func, int max_steps) {
    NeuralNetwork nn(std::vector<unsigned int>{
        (unsigned)GA_INPUTS, (unsigned)GA_HIDDEN, (unsigned)GA_OUTPUTS
    });

    // Cargar pesos embebidos
    std::vector<double> params(GA_PARAMS, GA_PARAMS + GA_PARAM_COUNT);
    nn.setParams(params);

    ale.reset_game();

    int lives = ale.lives();
    double total = 0.0;
    int step = 0;

    while (!ale.game_over() && step < max_steps) {
        // Misma heurística que usaste en training (si quieres consistencia)
        if (ale.lives() < lives) {
            lives = ale.lives();
            ale.act(PLAYER_A_FIRE);
        }

        auto ram = ale.getRAM();
        Matrix x(GA_INPUTS, 1);
        for (int i = 0; i < GA_INPUTS; ++i) {
            x.at(i, 0) = normalize_ram((uint8_t)ram.get(i));
        }

        Matrix out = nn.feedforward(x, activation_func);
        int a = argmax_colvec(out);

        reward_t r = ale.act(GA_ACTIONS[a]);
        total += (double)r;

        ++step;
    }

    return total;
}


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
///////////////////////////////////////////////////////////////////////////////

void usage(char const* pname) {
  std::cerr
    << "\nUSAGE:\n"
    << "  " << pname << " <romfile> heatmap <DSTfile>\n"
    << "  " << pname << " <romfile> dataset <DSTfile>\n"
    << "  " << pname << " <romfile> train <ORGfile> <DSTfile>\n"
    << "  " << pname << " <romfile> ga [max_steps]\n";
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
   bool gaMode      = (argc >= 3 && string(argv[2]) == "ga");

   
   // Check parameters and modes
   if (!trainMode && !datasetMode && !heatmapMode && !gaMode)
    usage(argv[0]);

   // Configure alei object.
   alei.setInt  ("random_seed", 0);
   alei.setFloat("repeat_action_probability", 0);
   alei.setBool ("display_screen", true);
   alei.setBool ("sound", true);
   alei.loadROM (argv[1]);
  
   if (gaMode) {
    int max_steps = 18000;
    int seed = 0;

    if (argc >= 4) max_steps = std::stoi(argv[3]);
    if (argc >= 5) seed = std::stoi(argv[4]);

    // Configuración para evaluación rápida y reproducible
    alei.setInt  ("random_seed", seed);
    alei.setFloat("repeat_action_probability", 0.0f);
    alei.setBool ("display_screen", true);   // pon false si quieres medir velocidad
    alei.setBool ("sound", false);

    // Cargar ROM una sola vez (aquí)
    alei.loadROM(argv[1]);

    double score = runGAMode(alei, "tanh", max_steps);
    std::cout << "GA score = " << score << "\n";
    return 0;
   }



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

      NeuralNetwork nn(std::vector<unsigned int>{
         128u, 64u, 32u, (unsigned)dataHelper.getOutputLayerSize()
      });

      nn.train(dataHelper.getInputs(), dataHelper.getTargets(),200, 0.01, "tanh");
      


   }


   


   return 0;
}
