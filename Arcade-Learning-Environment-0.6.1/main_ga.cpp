// main.cpp
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <limits>
#include <cstdint>

using namespace std;

#include "ga.h"
#include "ga_problems.h"

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

static bool parseIntArg(int& i, int argc, char** argv, int& out) {
    if (i + 1 >= argc) return false;
    out = atoi(argv[i + 1]);
    i++;
    return true;
}

static bool parseDoubleArg(int& i, int argc, char** argv, double& out) {
    if (i + 1 >= argc) return false;
    out = atof(argv[i + 1]);
    i++;
    return true;
}

static bool improvedEnough(Objective obj, double start, double end, double eps) {
    if (obj == Objective::Maximize) return end >= start + eps;
    return end <= start - eps;
}

// ------------------------------------------------------------
// Crear operadores según strings
// ------------------------------------------------------------

static unique_ptr<ISelection> makeSelection(const string& name) {
    if (name == "roulette") return make_unique<RouletteSelection>();
    return make_unique<TournamentSelection>();
}

static unique_ptr<ICrossover> makeCrossover(const string& name) {
    if (name == "uniform") return make_unique<UniformCrossover>();
    if (name == "blend")   return make_unique<BlendCrossover>();
    return make_unique<OnePointCrossover>();
}

static unique_ptr<IMutation> makeMutation(const string& name) {
    if (name == "gauss") return make_unique<GaussianMutation>();
    return make_unique<BitFlipMutation>();
}

// Default “razonable” según tipo de genoma
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
    int dim = 10;
    int bits = 100;

    // ALE (solo si lo ejecutas explícitamente con "ale")
    int ale_size = 50;
    int episodes = 1;
    int steps = 18000;

    // TSP
    int tsp_n = 25;
};

static void fillRandomCities(vector<TspRandomKeysProblem::Point2D>& cities, int n, GeneticAlgorithm& ga) {
    cities.clear();
    if (n <= 0) return;

    cities.resize((size_t)n);
    for (int i = 0; i < n; i++) {
        double x = ga.randomReal(0.0, 100.0);
        double y = ga.randomReal(0.0, 100.0);
        cities[(size_t)i].x = x;
        cities[(size_t)i].y = y;
    }
}

static shared_ptr<IProblem> makeProblem(const string& name,
                                        const ProblemOptions& po,
                                        GeneticAlgorithm& ga) {
    if (name == "sphere") {
        SphereProblem::Config cfg;
        cfg.dimension = (size_t)po.dim;
        auto p = make_shared<SphereProblem>(cfg);
        p->attachGA(&ga);
        return p;
    }

    if (name == "rastrigin") {
        RastriginProblem::Config cfg;
        cfg.dimension = (size_t)po.dim;
        auto p = make_shared<RastriginProblem>(cfg);
        p->attachGA(&ga);
        return p;
    }

    if (name == "ackley") {
        AckleyProblem::Config cfg;
        cfg.dimension = (size_t)po.dim;
        auto p = make_shared<AckleyProblem>(cfg);
        p->attachGA(&ga);
        return p;
    }

    if (name == "xor") {
        auto p = make_shared<XorProblem>();
        p->attachGA(&ga);
        return p;
    }

    if (name == "onemax") {
        OneMaxProblem::Config cfg;
        cfg.n_bits = (size_t)po.bits;
        auto p = make_shared<OneMaxProblem>(cfg);
        p->attachGA(&ga);
        return p;
    }

    if (name == "tsp") {
        TspRandomKeysProblem::Config cfg;
        cfg.use_distance_matrix = false;
        cfg.closed_tour = true;

        fillRandomCities(cfg.cities, po.tsp_n, ga);

        auto p = make_shared<TspRandomKeysProblem>(cfg);
        p->attachGA(&ga);
        return p;
    }

    if (name == "ale") {
        // IMPORTANTE:
        // Aquí sigue existiendo "ale", pero SOLO se ejecuta si llamas explícitamente:
        //   ./programa ale ...
        // En modo test, NO se incluye.

        AleAtariProblem::Config cfg;
        cfg.type = GenomeType::Real;
        cfg.genome_size = (size_t)po.ale_size;
        cfg.obj = Objective::Maximize;
        cfg.episodes_per_eval = po.episodes;
        cfg.max_steps_per_episode = po.steps;

        cfg.bounds.resize(cfg.genome_size);
        for (size_t i = 0; i < cfg.genome_size; i++) {
            cfg.bounds[i].lo = -1.0;
            cfg.bounds[i].hi =  1.0;
        }

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

        cfg.fitness_fn = [](const AleAtariProblem& self, const Genome& g) {
            if (g.getType() != GenomeType::Real) return -1e9;

            double sum = 0.0;
            for (double x : g.reals) sum += x * x;

            int eps = self.getEpisodesPerEval();
            int stp = self.getMaxStepsPerEpisode();
            double scale = 1.0 + 0.000001 * (double)(eps + stp);

            return (1.0 / (1.0 + sum)) / scale;
        };

        cfg.describe_fn = [](const AleAtariProblem& self, const Genome& g) {
            string s = "ALE(dummy) size=" + to_string(self.genomeSize());
            if (g.getType() == GenomeType::Real && !g.reals.empty()) {
                s += " x0=" + to_string(g.reals[0]);
            }
            return s;
        };

        auto p = make_shared<AleAtariProblem>(cfg);
        p->attachGA(&ga);
        return p;
    }

    return nullptr;
}

// ------------------------------------------------------------
// Ejecutar una corrida normal
// ------------------------------------------------------------

static int runProblem(const string& problem_name,
                      GeneticAlgorithm::Config ga_cfg,
                      const ProblemOptions& po,
                      const string& sel_name,
                      const string& cx_name,
                      const string& mx_name,
                      bool use_custom_ops) {
    GeneticAlgorithm ga;
    ga.setConfig(ga_cfg);
    ga.setSeed(ga_cfg.seed);

    auto prob = makeProblem(problem_name, po, ga);
    if (!prob) {
        cerr << "Unknown problem: " << problem_name << endl;
        return 1;
    }

    ga.setProblem(prob);

    if (use_custom_ops) {
        ga.setSelection(makeSelection(sel_name));
        ga.setCrossover(makeCrossover(cx_name));
        ga.setMutation(makeMutation(mx_name));
    } else {
        setDefaultOperators(ga, prob->genomeType());
    }

    ga.setGenerationCallback([](const GeneticAlgorithm& g, const GeneticAlgorithm::GenerationStats& st) {
        if (st.generation % 10 == 0) {
            cout << "gen " << st.generation
                 << " best=" << st.best_fitness
                 << " mean=" << st.mean_fitness
                 << " worst=" << st.worst_fitness << endl;
        }
    });

    ga.initialize();
    while (!ga.step()) {}

    const Individual& best = ga.getBestIndividual();
    cout << "\nDONE\n";
    cout << "best fitness: " << best.fitness << endl;
    cout << "best genome:  " << prob->describe(best.genome) << endl;
    cout << "generations:  " << ga.getGeneration() << endl;

    return 0;
}

// ------------------------------------------------------------
// Modo TEST: muchas pruebas
// ------------------------------------------------------------

struct OpCombo {
    string sel;
    string cx;
    string mx;
};

static void applyCombo(GeneticAlgorithm& ga, const OpCombo& c) {
    ga.setSelection(makeSelection(c.sel));
    ga.setCrossover(makeCrossover(c.cx));
    ga.setMutation(makeMutation(c.mx));
}

static int runTests(GeneticAlgorithm::Config base_cfg, const ProblemOptions& po) {
    vector<OpCombo> combos = {
        {"tournament", "blend",   "gauss"},
        {"roulette",   "blend",   "gauss"},
        {"tournament", "uniform", "gauss"},
        {"tournament", "onepoint","bitflip"},
        {"roulette",   "uniform", "bitflip"}
    };

    // IMPORTANTE: ale NO entra en la batería de tests
    vector<string> problems = {"sphere", "rastrigin", "ackley", "xor", "onemax", "tsp"};

    int trials_per_case = 8;
    double eps_improve = 1e-9;

    int total_cases = 0;
    int cases_with_improvement = 0;
    int cases_reached_target = 0;

    cout << "TEST MODE\n";
    cout << "Trials per case: " << trials_per_case << endl;
    cout << "Combos: " << combos.size() << endl;
    cout << "Problems: " << problems.size() << endl << endl;

    for (const string& pname : problems) {
        cout << "== Problem: " << pname << " ==\n";

        for (int cidx = 0; cidx < (int)combos.size(); cidx++) {
            const OpCombo& combo = combos[(size_t)cidx];

            int improved_count = 0;
            int target_count = 0;

            for (int t = 0; t < trials_per_case; t++) {
                GeneticAlgorithm ga;

                GeneticAlgorithm::Config cfg = base_cfg;

                uint32_t extra = (uint32_t)(pname.size() * 1000 + cidx * 100 + t * 7);
                cfg.seed = base_cfg.seed + extra;

                bool base_target_defined = (cfg.stop.target_fitness == cfg.stop.target_fitness);
                if (!base_target_defined) {
                    if (pname == "xor") cfg.stop.target_fitness = 4.0;
                    if (pname == "onemax") cfg.stop.target_fitness = (double)po.bits;
                }

                ga.setConfig(cfg);
                ga.setSeed(cfg.seed);

                auto prob = makeProblem(pname, po, ga);
                if (!prob) continue;

                ga.setProblem(prob);
                applyCombo(ga, combo);

                ga.initialize();
                double start_best = ga.getBestIndividual().fitness;

                while (!ga.step()) {}

                double end_best = ga.getBestIndividual().fitness;

                Objective obj = prob->objective();
                bool improved = improvedEnough(obj, start_best, end_best, eps_improve);

                if (improved) improved_count++;

                double target = cfg.stop.target_fitness;
                bool target_is_defined = (target == target);
                if (target_is_defined) {
                    bool reached = false;
                    if (obj == Objective::Maximize) reached = (end_best >= target);
                    else reached = (end_best <= target);
                    if (reached) target_count++;
                }

                total_cases++;
                if (improved) cases_with_improvement++;
            }

            cases_reached_target += target_count;

            cout << " combo sel=" << combo.sel
                 << " cx=" << combo.cx
                 << " mx=" << combo.mx
                 << " | improved " << improved_count << "/" << trials_per_case
                 << " | reached_target " << target_count << "/" << trials_per_case
                 << "\n";
        }

        cout << endl;
    }

    cout << "SUMMARY\n";
    cout << " total trials:         " << total_cases << endl;
    cout << " improved trials:      " << cases_with_improvement << endl;
    cout << " reached target trials:" << cases_reached_target << endl;

    return 0;
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        printHelp(argv[0]);
        return 0;
    }

    string mode_or_problem = argv[1];
    if (mode_or_problem == "--help" || mode_or_problem == "-h") {
        printHelp(argv[0]);
        return 0;
    }

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

    ProblemOptions po;

    bool use_custom_ops = false;
    string sel_name = "tournament";
    string cx_name = "blend";
    string mx_name = "gauss";

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

        // ALE solo si se ejecuta explícitamente
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

    if (mode_or_problem == "test" || mode_or_problem == "--test") {
        return runTests(ga_cfg, po);
    }

    string problem_name = mode_or_problem;

    bool target_is_defined = (ga_cfg.stop.target_fitness == ga_cfg.stop.target_fitness);
    if (!target_is_defined) {
        if (problem_name == "xor") ga_cfg.stop.target_fitness = 4.0;
        if (problem_name == "onemax") ga_cfg.stop.target_fitness = (double)po.bits;
    }

    return runProblem(problem_name, ga_cfg, po, sel_name, cx_name, mx_name, use_custom_ops);
}
