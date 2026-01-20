// main.cpp
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <limits>
#include <cstdint>

using namespace std;

#include "new_ga.h"
#include "problemas_ga.h"

using namespace ga;

// ------------------------------------------------------------
// Helpers simples
// ------------------------------------------------------------

static void printHelp(const char* exe) {
    cout << "Usage:\n";
    cout << "  " << exe << " <problem> [options]\n";
    cout << "  " << exe << " test [options]\n\n";

    cout << "Problems:\n";
    cout << "  sphere | rastrigin | ackley | xor | onemax | ale | tsp\n";
    cout << "  (NOTE: test mode does NOT run ale)\n\n";

    cout << "Options:\n";
    cout << "  --pop N          population size (default 50)\n";
    cout << "  --gen N          max generations (default 200)\n";
    cout << "  --seed N         seed (default 123)\n";
    cout << "  --elite N        elitism count (default 1)\n";
    cout << "  --cross P        crossover rate (default 0.9)\n";
    cout << "  --mut P          mutation rate (default 0.2)\n";
    cout << "  --target X       target fitness (optional)\n\n";

    cout << "Operators:\n";
    cout << "  --sel tournament|roulette\n";
    cout << "  --cx  onepoint|uniform|blend\n";
    cout << "  --mx  bitflip|gauss\n\n";

    cout << "Problem-specific:\n";
    cout << "  --dim N          (sphere/rastrigin/ackley) dimension\n";
    cout << "  --bits N         (onemax) number of bits\n";
    cout << "  --ale-size N     (ale) genome size (dummy weights)\n";
    cout << "  --episodes N     (ale) episodes per eval\n";
    cout << "  --steps N        (ale) max steps per episode\n";
    cout << "  --tsp-n N        (tsp) number of cities (default 25)\n\n";

    cout << "Examples:\n";
    cout << "  " << exe << " sphere --dim 20 --pop 80 --gen 300\n";
    cout << "  " << exe << " xor --pop 40 --gen 80 --target 4\n";
    cout << "  " << exe << " tsp --tsp-n 40 --pop 80 --gen 400\n";
    cout << "  " << exe << " ale --ale-size 60 --episodes 3 --steps 5000\n";
    cout << "  " << exe << " test --gen 150\n";
}
//lee un entero que viene despues de una opcion(en los argumentos)
static bool parseIntArg(int& i, int argc, char** argv, int& out) {
    //miro si existe un siguiente argumento
    if (i + 1 >= argc) return false;
    //texto a int
    out = atoi(argv[i + 1]);
    //salto al nuevo valor
    i++;
    return true;
}
//lo mismo que lo anterior pero a float
static bool parseDoubleArg(int& i, int argc, char** argv, double& out) {
    if (i + 1 >= argc) return false;
    out = atof(argv[i + 1]);
    i++;
    return true;
}
//decide si ha mejorado lo suficiente el mejor fitness tras entrenar
static bool improvedEnough(Objective obj, double start, double end, double eps) {
    //si es maximizar  mejorar es que end sea mayor que start +eps(epsilon, margen minimo de mejora)
    if (obj == Objective::Maximize) return end >= start + eps;
    return end <= start - eps;
}

// ------------------------------------------------------------
// Crear operadores según strings
// ------------------------------------------------------------
//elige que metodo usa el GA segun nombre
static unique_ptr<ISelection> makeSelection(const string& name) {
    if (name == "roulette") return make_unique<RouletteSelection>();
    return make_unique<TournamentSelection>();
}
//elige que tipo de cruce usar
static unique_ptr<ICrossover> makeCrossover(const string& name) {
    if (name == "uniform") return make_unique<UniformCrossover>();
    if (name == "blend")   return make_unique<BlendCrossover>();
    return make_unique<OnePointCrossover>();
}
//elige que tipo de mutacion usar
static unique_ptr<IMutation> makeMutation(const string& name) {
    if (name == "gauss") return make_unique<GaussianMutation>();
    return make_unique<BitFlipMutation>();
}

// Si el usuario no pone argumentos, se crea esto como default dependiendo si es binario o no el problema
static void setDefaultOperators(GeneticAlgorithm& ga, GenomeType t) {
    ga.setSelection(make_unique<TournamentSelection>());

    if (t == GenomeType::Binary) {
        ga.setCrossover(make_unique<OnePointCrossover>());
        ga.setMutation(make_unique<BitFlipMutation>());
    } else {
        ga.setCrossover(make_unique<BlendCrossover>());
        ga.setMutation(make_unique<GaussianMutation>());
    }
}

// ------------------------------------------------------------
// Crear problemas según nombre + opciones
// ------------------------------------------------------------

struct ProblemOptions {
    int dim = 10;//variables de la solucion(SPHERE, RASTRIGIN y ACKLEY)
    int bits = 100;//bits del problema ONEMAX

    
    int ale_size = 50;//tamaño genoma problema ale
    int episodes = 1;//partidas o eopisodios para evaluar individuo
    int steps = 18000;//es un max steps

    // TSP
    int tsp_n = 25;//numero de ciudades en problema del viajante ante
};
//cities es el vector lista de ciudades
static void fillRandomCities(vector<TspRandomKeysProblem::Point2D>& cities, int n, GeneticAlgorithm& ga) {
    cities.clear();
    if (n <= 0) return;
    //si las pedidas son 0 no hago nada, sino resizeo
    cities.resize((size_t)n);
    for (int i = 0; i < n; i++) {
        //genero x al azar y un y al azar y los guardo en la ciudad i
        double x = ga.randomReal(0.0, 100.0);
        double y = ga.randomReal(0.0, 100.0);
        cities[(size_t)i].x = x;
        cities[(size_t)i].y = y;
    }
}
//esta funcion es LOCURA de larga , doy nombre de problema y me devuelve objeto implementando la interfaz iproblem
static shared_ptr<IProblem> makeProblem(const string& name,
                                        const ProblemOptions& po,
                                        GeneticAlgorithm& ga) {
    if (name == "sphere") {
        SphereProblem::Config cfg;
        cfg.dimension = (size_t)po.dim;
        auto p = make_shared<SphereProblem>(cfg);//creo problema
        p->attachGA(&ga);//lo conecto con el ga
        return p;
    }
    //mismo que sphere
    if (name == "rastrigin") {
        RastriginProblem::Config cfg;
        cfg.dimension = (size_t)po.dim;
        auto p = make_shared<RastriginProblem>(cfg);
        p->attachGA(&ga);
        return p;
    }
    //mismo que sphere
    if (name == "ackley") {
        AckleyProblem::Config cfg;
        cfg.dimension = (size_t)po.dim;
        auto p = make_shared<AckleyProblem>(cfg);
        p->attachGA(&ga);
        return p;
    }
    //no necesito config
    if (name == "xor") {
        auto p = make_shared<XorProblem>();
        p->attachGA(&ga);
        return p;
    }
    //decido cuantos bits tendra el genoma
    if (name == "onemax") {
        OneMaxProblem::Config cfg;
        cfg.n_bits = (size_t)po.bits;
        auto p = make_shared<OneMaxProblem>(cfg);
        p->attachGA(&ga);
        return p;
    }

    if (name == "tsp") {
        TspRandomKeysProblem::Config cfg;
        cfg.use_distance_matrix = false;//se hace matriz?
        cfg.closed_tour = true;//decido si tour vuelve al inicio o no

        fillRandomCities(cfg.cities, po.tsp_n, ga);//genero ciudades aleatorias

        auto p = make_shared<TspRandomKeysProblem>(cfg);
        p->attachGA(&ga);
        return p;
    }

    if (name == "ale") {
       
        AleAtariProblem::Config cfg;
        cfg.type = GenomeType::Real;
        //genoma real tamaño po ale size
        cfg.genome_size = (size_t)po.ale_size;
        cfg.obj = Objective::Maximize;
        cfg.episodes_per_eval = po.episodes;
        cfg.max_steps_per_episode = po.steps;
        //bounds de -1 y +1 para cada gen
        cfg.bounds.resize(cfg.genome_size);
        for (size_t i = 0; i < cfg.genome_size; i++) {
            cfg.bounds[i].lo = -1.0;
            cfg.bounds[i].hi =  1.0;
        }
        //crear genoma iniciar aleatorio
        cfg.init_fn = [](const AleAtariProblem& self, Genome& out_genome) {
            const GeneticAlgorithm* ga_ptr = self.getAttachedGA();
            out_genome.setType(GenomeType::Real);
            out_genome.resizeReal(self.genomeSize());

            vector<Bounds> b;
            self.getBounds(b);

            for (size_t i = 0; i < self.genomeSize(); i++) {
                double v = 0.0;
                if (ga_ptr) v = ga_ptr->randomReal(b[i].lo, b[i].hi);
                else v = b[i].lo + (b[i].hi - b[i].lo) * ((double)rand() / RAND_MAX);
                out_genome.setReal(i, v);
            }
        };
        //calcular fitness
        cfg.fitness_fn = [](const AleAtariProblem& self, const Genome& g) {
            if (g.getType() != GenomeType::Real) return -1e9;

            double sum = 0.0;
            for (double x : g.reals) sum += x * x;

            int eps = self.getEpisodesPerEval();
            int stp = self.getMaxStepsPerEpisode();
            double scale = 1.0 + 0.000001 * (double)(eps + stp);

            return (1.0 / (1.0 + sum)) / scale;
        };
        //logs
        cfg.describe_fn = [](const AleAtariProblem& self, const Genome& g) {
            string s = "ALE(dummy) size=" + to_string(self.genomeSize());
            if (g.getType() == GenomeType::Real && !g.reals.empty()) {
                s += " x0=" + to_string(g.reals[0]);
            }
            return s;
        };
        //
        auto p = make_shared<AleAtariProblem>(cfg);
        p->attachGA(&ga);
        return p;
    }

    return nullptr;
}

//i am running booys
//po es dim bits tsp_n...
//sel cx mx son nombres de operadores
//custom ops true si hay q usar los nombres pasados y false si no
static int runProblem(const string& problem_name,
                      GeneticAlgorithm::Config ga_cfg,
                      const ProblemOptions& po,
                      const string& sel_name,
                      const string& cx_name,
                      const string& mx_name,
                      bool use_custom_ops) {
    //seteamos cosicas, no tiene mucho misterio                    
    GeneticAlgorithm ga;
    ga.setConfig(ga_cfg);
    ga.setSeed(ga_cfg.seed);
    //creo problema y si no existe devuelvo error                  
    auto prob = makeProblem(problem_name, po, ga);
    if (!prob) {
        cerr << "Unknown problem: " << problem_name << endl;
        return 1;
    }
    //setter
    ga.setProblem(prob);
    //decido que operadores usar
    if (use_custom_ops) {
        ga.setSelection(makeSelection(sel_name));
        ga.setCrossover(makeCrossover(cx_name));
        ga.setMutation(makeMutation(mx_name));
    } else {
        setDefaultOperators(ga, prob->genomeType());
    }
    //imprimir progreso
    ga.setGenerationCallback([](const GeneticAlgorithm& g, const GeneticAlgorithm::GenerationStats& st) {
        if (st.generation % 10 == 0) {
            cout << "gen " << st.generation
                 << " best=" << st.best_fitness
                 << " mean=" << st.mean_fitness
                 << " worst=" << st.worst_fitness << endl;
        }
    });

    ga.initialize();
    //sigues haciendo generaciones hasta que acabe step, step hace una gen
    while (!ga.step()) {}
    //logs
    const Individual& best = ga.getBestIndividual();
    cout << "\nDONE\n";
    cout << "best fitness: " << best.fitness << endl;
    cout << "best genome:  " << prob->describe(best.genome) << endl;
    cout << "generations:  " << ga.getGeneration() << endl;

    return 0;
}
//testing melting
struct OpCombo {
    string sel;//seleccion tournament o roulette
    string cx;//cruce blend uniform onepoint
    string mx;//mutacion gauss o bitflip
};

static void applyCombo(GeneticAlgorithm& ga, const OpCombo& c) {
    //cojo c y lo aplico en el ga
    ga.setSelection(makeSelection(c.sel));
    ga.setCrossover(makeCrossover(c.cx));
    ga.setMutation(makeMutation(c.mx));
}
//base cfg es la config base del ga, como una plantilla
//po opciones de problemas
static int runTests(GeneticAlgorithm::Config base_cfg, const ProblemOptions& po) {
    //lista de combos a probar, la idea es ver cual funciona mejor
    vector<OpCombo> combos = {
        {"tournament", "blend",   "gauss"},
        {"roulette",   "blend",   "gauss"},
        {"tournament", "uniform", "gauss"},
        {"tournament", "onepoint","bitflip"},
        {"roulette",   "uniform", "bitflip"}
    };

    //problemas a testear, todos menos ale
    vector<string> problems = {"sphere", "rastrigin", "ackley", "xor", "onemax", "tsp"};
    //cada caso se repite 8 veces con seeds distintas
    int trials_per_case = 8;
    //minimo para contar como mejora
    double eps_improve = 1e-9;
    //contadores globales de tests,ejecuciones totales cuantas mejraron y cuantas alcanzaron el target
    int total_cases = 0;
    int cases_with_improvement = 0;
    int cases_reached_target = 0;
    //un poco de cabeceras para que quede mejor jeje
    cout << "TEST MODE\n";
    cout << "Trials per case: " << trials_per_case << endl;
    cout << "Combos: " << combos.size() << endl;
    cout << "Problems: " << problems.size() << endl << endl;
    //para cada problema
    for (const string& pname : problems) {
        cout << "== Problem: " << pname << " ==\n";
        //por cada combinacion de operadores
        for (int cidx = 0; cidx < (int)combos.size(); cidx++) {
            const OpCombo& combo = combos[(size_t)cidx];
            //voy probando cada combo y veo cuanto mejoro(improved count) y cuantas veces llego al target(target count)
            int improved_count = 0;
            int target_count = 0;
            //repito 8 veces(trials) con seeds distintas
            for (int t = 0; t < trials_per_case; t++) {
                GeneticAlgorithm ga;

                GeneticAlgorithm::Config cfg = base_cfg;
                uint32_t extra = (uint32_t)(pname.size() * 1000 + cidx * 100 + t * 7);
                cfg.seed = base_cfg.seed + extra;
                //si no hay targett definido pongo defaults
                bool base_target_defined = (cfg.stop.target_fitness == cfg.stop.target_fitness);
                if (!base_target_defined) {
                    if (pname == "xor") cfg.stop.target_fitness = 4.0;//fit max
                    if (pname == "onemax") cfg.stop.target_fitness = (double)po.bits;//num bits
                }
                //setters
                ga.setConfig(cfg);
                ga.setSeed(cfg.seed);
                //construyo problema y aplico operadores
                auto prob = makeProblem(pname, po, ga);
                if (!prob) continue;

                ga.setProblem(prob);
                applyCombo(ga, combo);
                //ejecuto y mido mejora
                ga.initialize();
                double start_best = ga.getBestIndividual().fitness;//mejor fitness al principio

                while (!ga.step()) {}

                double end_best = ga.getBestIndividual().fitness;//mejor fitness final
                //decido si mejoro lo suficiente, objective me dice si es subir o bajar
                Objective obj = prob->objective();
                bool improved = improvedEnough(obj, start_best, end_best, eps_improve);

                if (improved) improved_count++;
                //compruebo si se alcanzo el target
                double target = cfg.stop.target_fitness;
                bool target_is_defined = (target == target);
                if (target_is_defined) {
                    bool reached = false;
                    if (obj == Objective::Maximize) reached = (end_best >= target);//maximize
                    else reached = (end_best <= target);//minimice
                    if (reached) target_count++;
                }

                total_cases++;
                if (improved) cases_with_improvement++;
            }

            cases_reached_target += target_count;
            //prints chorras
            cout << " combo sel=" << combo.sel
                 << " cx=" << combo.cx
                 << " mx=" << combo.mx
                 << " | improved " << improved_count << "/" << trials_per_case
                 << " | reached_target " << target_count << "/" << trials_per_case
                 << "\n";
        }

        cout << endl;
    }
    //sumario
    cout << "SUMMARY\n";
    cout << " total trials:         " << total_cases << endl;
    cout << " improved trials:      " << cases_with_improvement << endl;
    cout << " reached target trials:" << cases_reached_target << endl;

    return 0;
}



int main(int argc, char** argv) {
    //si no hay nada printeo el usage de toda la vida
    if (argc < 2) {
        printHelp(argv[0]);
        return 0;
    }
    //guardo primer arg
    string mode_or_problem = argv[1];
    if (mode_or_problem == "--help" || mode_or_problem == "-h") {
        printHelp(argv[0]);
        return 0;
    }
//config por defecto del ga
    GeneticAlgorithm::Config ga_cfg;
    ga_cfg.population_size = 50;
    ga_cfg.stop.max_generations = 200;
    ga_cfg.seed = 123;
    ga_cfg.elitism_count = 1;
    ga_cfg.crossover_rate = 0.9;
    ga_cfg.mutation_rate = 0.2;

    ga_cfg.stop.max_stagnant_generations = 50;
    ga_cfg.stop.min_delta = 1e-12;
    ga_cfg.stop.target_fitness = numeric_limits<double>::quiet_NaN();
    //opciones y operadores por defecto
    ProblemOptions po;

    bool use_custom_ops = false;
    string sel_name = "tournament";
    string cx_name = "blend";
    string mx_name = "gauss";
    //empiezo a leer args
    /*
    --pop N → cambia ga_cfg.population_size

    --gen N → cambia ga_cfg.stop.max_generations

    --seed N → cambia ga_cfg.seed

    --elite N → cambia ga_cfg.elitism_count

    --cross P → cambia ga_cfg.crossover_rate

    --mut P → cambia ga_cfg.mutation_rate

    --target X → cambia ga_cfg.stop.target_fitness

    Uso parseIntArg y parseDoubleArg para:(si no se usa es que es texto)

    comprobar que hay un valor después

    convertirlo a int/double

    avanzar i

    Si falta el valor, imprimo error y return 1
    --dim N → po.dim (sphere/rastrigin/ackley)

    --bits N → po.bits (onemax)

    --ale-size N, --episodes N, --steps N → opciones ale

    --tsp-n N → po.tsp_n
    */
    for (int i = 2; i < argc; i++) {
        string a = argv[i];

        if (a == "--pop") {
            int v; if (!parseIntArg(i, argc, argv, v)) { cerr << "Missing value for --pop\n"; return 1; }
            ga_cfg.population_size = (size_t)v;
            continue;
        }
        if (a == "--gen") {
            int v; if (!parseIntArg(i, argc, argv, v)) { cerr << "Missing value for --gen\n"; return 1; }
            ga_cfg.stop.max_generations = (size_t)v;
            continue;
        }
        if (a == "--seed") {
            int v; if (!parseIntArg(i, argc, argv, v)) { cerr << "Missing value for --seed\n"; return 1; }
            ga_cfg.seed = (uint32_t)v;
            continue;
        }
        if (a == "--elite") {
            int v; if (!parseIntArg(i, argc, argv, v)) { cerr << "Missing value for --elite\n"; return 1; }
            ga_cfg.elitism_count = (size_t)v;
            continue;
        }
        if (a == "--cross") {
            double v; if (!parseDoubleArg(i, argc, argv, v)) { cerr << "Missing value for --cross\n"; return 1; }
            ga_cfg.crossover_rate = v;
            continue;
        }
        if (a == "--mut") {
            double v; if (!parseDoubleArg(i, argc, argv, v)) { cerr << "Missing value for --mut\n"; return 1; }
            ga_cfg.mutation_rate = v;
            continue;
        }
        if (a == "--target") {
            double v; if (!parseDoubleArg(i, argc, argv, v)) { cerr << "Missing value for --target\n"; return 1; }
            ga_cfg.stop.target_fitness = v;
            continue;
        }

        if (a == "--sel") {
            if (i + 1 >= argc) { cerr << "Missing value for --sel\n"; return 1; }
            sel_name = argv[i + 1];
            use_custom_ops = true;
            i++;
            continue;
        }
        if (a == "--cx") {
            if (i + 1 >= argc) { cerr << "Missing value for --cx\n"; return 1; }
            cx_name = argv[i + 1];
            use_custom_ops = true;
            i++;
            continue;
        }
        if (a == "--mx") {
            if (i + 1 >= argc) { cerr << "Missing value for --mx\n"; return 1; }
            mx_name = argv[i + 1];
            use_custom_ops = true;
            i++;
            continue;
        }

        if (a == "--dim") {
            int v; if (!parseIntArg(i, argc, argv, v)) { cerr << "Missing value for --dim\n"; return 1; }
            po.dim = v;
            continue;
        }
        if (a == "--bits") {
            int v; if (!parseIntArg(i, argc, argv, v)) { cerr << "Missing value for --bits\n"; return 1; }
            po.bits = v;
            continue;
        }

        if (a == "--ale-size") {
            int v; if (!parseIntArg(i, argc, argv, v)) { cerr << "Missing value for --ale-size\n"; return 1; }
            po.ale_size = v;
            continue;
        }
        if (a == "--episodes") {
            int v; if (!parseIntArg(i, argc, argv, v)) { cerr << "Missing value for --episodes\n"; return 1; }
            po.episodes = v;
            continue;
        }
        if (a == "--steps") {
            int v; if (!parseIntArg(i, argc, argv, v)) { cerr << "Missing value for --steps\n"; return 1; }
            po.steps = v;
            continue;
        }

        if (a == "--tsp-n") {
            int v; if (!parseIntArg(i, argc, argv, v)) { cerr << "Missing value for --tsp-n\n"; return 1; }
            po.tsp_n = v;
            continue;
        }

        cerr << "Unknown option: " << a << endl;
        return 1;
    }
    //si el primero era test ejecuto run tests
    if (mode_or_problem == "test" || mode_or_problem == "--test") {
        return runTests(ga_cfg, po);
    }
    //sino, era un problema
    string problem_name = mode_or_problem;
    //y miro si hay targets definidos o no y si no hay pongo los defaults
    bool target_is_defined = (ga_cfg.stop.target_fitness == ga_cfg.stop.target_fitness);
    if (!target_is_defined) {
        if (problem_name == "xor") ga_cfg.stop.target_fitness = 4.0;
        if (problem_name == "onemax") ga_cfg.stop.target_fitness = (double)po.bits;
    }

    return runProblem(problem_name, ga_cfg, po, sel_name, cx_name, mx_name, use_custom_ops);
}
