// rcga_train_ale.cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "src/ale_interface.hpp" // igual que minimal_agent.cpp
#include "ga.cpp"                // TU RealCodedGA (sí: se incluye como "header")

static constexpr int RAM_BYTES = 128;

// -------------------- Guardado / Carga de pesos --------------------

static void save_genome_txt(const std::string& path, int hidden, int nActions,
                            const std::vector<double>& genome) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("No puedo abrir para escribir: " + path);

    f << "hidden "  << hidden  << "\n";
    f << "actions " << nActions << "\n";
    f << "dim "     << genome.size() << "\n";
    f << std::setprecision(17);
    for (double w : genome) f << w << "\n";
}

static std::vector<double> load_genome_txt(const std::string& path,
                                           int& hidden_out, int& nActions_out) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("No puedo abrir para leer: " + path);

    std::string tag;
    std::size_t dim = 0;

    f >> tag >> hidden_out;   if (tag != "hidden")  throw std::runtime_error("Formato inválido (hidden)");
    f >> tag >> nActions_out; if (tag != "actions") throw std::runtime_error("Formato inválido (actions)");
    f >> tag >> dim;          if (tag != "dim")     throw std::runtime_error("Formato inválido (dim)");

    std::vector<double> g(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        if (!(f >> g[i])) throw std::runtime_error("Faltan pesos en: " + path);
    }
    return g;
}

// Header para #include (requiere recompilar para usarlo)
static void save_genome_header(const std::string& path, int hidden, int nActions,
                               const std::vector<double>& genome) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("No puedo abrir para escribir: " + path);

    f << "#pragma once\n";
    f << "// Auto-generado por rcga_train_ale\n";
    f << "static constexpr int BEST_HIDDEN  = " << hidden << ";\n";
    f << "static constexpr int BEST_ACTIONS = " << nActions << ";\n";
    f << "static constexpr int BEST_DIM     = " << genome.size() << ";\n";
    f << "static const double BEST_GENOME[BEST_DIM] = {\n";
    f << std::setprecision(17);

    for (std::size_t i = 0; i < genome.size(); ++i) {
        f << "  " << genome[i] << (i + 1 < genome.size() ? "," : "") << "\n";
    }
    f << "};\n";
}

// -------------------- MLP policy (RAM -> logits -> argmax action) --------------------

struct MLPPolicy {
    int in = RAM_BYTES;
    int hidden = 32;
    int out = 0; // actions.size()

    // genoma = [W1(hidden*in), b1(hidden), W2(out*hidden), b2(out)]
    const std::vector<double>* g = nullptr;

    mutable std::vector<double> h; // hidden activations
    mutable std::vector<double> o; // output logits

    static std::size_t genome_size(int in, int hidden, int out) {
        return (std::size_t)hidden * (std::size_t)in + (std::size_t)hidden
             + (std::size_t)out    * (std::size_t)hidden + (std::size_t)out;
    }

    void bind(const std::vector<double>& genome, int hidden_, int out_) {
        g = &genome;
        hidden = hidden_;
        out = out_;
        h.assign((std::size_t)hidden, 0.0);
        o.assign((std::size_t)out, 0.0);
    }

    // Normaliza RAM byte -> [-1, 1]
    static inline double norm_byte(uint8_t v) {
        return (double(v) / 255.0) * 2.0 - 1.0;
    }

    // Devuelve índice [0..out-1]
    template <class RamT>
    int act_index(const RamT& ram) const {
        const auto& G = *g;
        std::size_t off = 0;

        const double* W1 = G.data() + off; off += (std::size_t)hidden * (std::size_t)in;
        const double* b1 = G.data() + off; off += (std::size_t)hidden;
        const double* W2 = G.data() + off; off += (std::size_t)out * (std::size_t)hidden;
        const double* b2 = G.data() + off; off += (std::size_t)out;

        // hidden = tanh(W1*x + b1)
        for (int j = 0; j < hidden; ++j) {
            double s = b1[j];
            const std::size_t row = (std::size_t)j * (std::size_t)in;
            for (int i = 0; i < in; ++i) {
                const double xi = norm_byte((uint8_t)ram.get(i));
                s += W1[row + (std::size_t)i] * xi;
            }
            h[(std::size_t)j] = std::tanh(s);
        }

        // out logits = W2*h + b2
        int bestk = 0;
        double bestv = -1e300;
        for (int k = 0; k < out; ++k) {
            double s = b2[k];
            const std::size_t row = (std::size_t)k * (std::size_t)hidden;
            for (int j = 0; j < hidden; ++j) {
                s += W2[row + (std::size_t)j] * h[(std::size_t)j];
            }
            o[(std::size_t)k] = s;
            if (s > bestv) { bestv = s; bestk = k; }
        }
        return bestk;
    }
};

// -------------------- Evaluador ALE --------------------

class AleEvaluator {
public:
    AleEvaluator(const std::string& rom,
                 int seed,
                 bool render,
                 int maxSteps)
    : rom_(rom), seed_(seed), render_(render), maxSteps_(maxSteps) {

        ale_.setInt("random_seed", seed_);
        ale_.setFloat("repeat_action_probability", 0.0f);
        ale_.setBool("display_screen", render_);
        ale_.setBool("sound", render_);

        ale_.loadROM(rom_);

        actions_ = ale_.getMinimalActionSet();
        if (actions_.empty()) {
            throw std::runtime_error("ALE: minimal action set empty");
        }

        // Estado inicial
        ale_.reset_game();

        // Si tu versión de ALE no tiene clone/restore, comenta estas líneas
        // y usa reset_game() dentro de rollout().
        startState_ = ale_.cloneSystemState();

        hasFire_ = (std::find(actions_.begin(), actions_.end(), PLAYER_A_FIRE) != actions_.end());
    }

    const std::vector<Action>& actions() const { return actions_; }

    double rollout(const std::vector<double>& genome, int hidden, int episodes) {
        MLPPolicy pol;
        pol.bind(genome, hidden, (int)actions_.size());

        double total = 0.0;

        for (int ep = 0; ep < episodes; ++ep) {
            // Igualar inicio para reducir ruido del fitness
            ale_.restoreSystemState(startState_);
            // Si no tienes restoreSystemState:
            // ale_.reset_game();

            // Arranque típico
            if (hasFire_) {
                for (int i = 0; i < 3; ++i) ale_.act(PLAYER_A_FIRE);
            }

            int step = 0;
            while (!ale_.game_over() && step < maxSteps_) {
                auto ram = ale_.getRAM();

                int k = pol.act_index(ram);
                if (k < 0) k = 0;
                if (k >= (int)actions_.size()) k = (int)actions_.size() - 1;

                reward_t r = ale_.act(actions_[(std::size_t)k]);
                total += (double)r;
                ++step;
            }
        }

        return total / std::max(1, episodes);
    }

private:
    std::string rom_;
    int seed_ = 0;
    bool render_ = false;
    int maxSteps_ = 500;

    ALEInterface ale_;
    std::vector<Action> actions_;
    bool hasFire_ = false;

    ALEState startState_;
};

// -------------------- MAIN --------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
          << "USO:\n"
          << "  " << argv[0] << " <romfile> train [gens=200] [pop=100] [steps=500] [episodes=1] [hidden=32]\n"
          << "       [save_header=best_genome.hpp] [save_txt=best_genome.txt]\n"
          << "  " << argv[0] << " <romfile> play  [steps=20000] [hidden=32] [load_txt=best_genome.txt]\n"
          << "\n"
          << "Compatibilidad (modo antiguo):\n"
          << "  " << argv[0] << " <romfile> [gens pop steps episodes hidden]\n"
          << "    (equivale a 'train' con nombres por defecto)\n";
        return 1;
    }

    const std::string rom = argv[1];
    std::string mode = "train";
    int argi = 2;

    // Si el 2º argumento es modo explícito:
    if (argc >= 3) {
        std::string m = argv[2];
        if (m == "train" || m == "play" || m == "display") {
            mode = (m == "display") ? "play" : m;
            argi = 3;
        }
    }

    auto get_i = [&](int def) -> int {
    if (argi < argc) {
        try {
            return std::stoi(argv[argi++]);
        } catch (const std::exception&) {
            throw std::runtime_error(std::string("Argumento entero inválido: '") + argv[argi] + "'");
        }
    }
    return def;
};

    auto get_s = [&](const std::string& def) -> std::string {
        if (argi < argc) return std::string(argv[argi++]);
        return def;
    };

    // ---------------- PLAY ----------------
    if (mode == "play") {
        const int steps  = get_i(20000);
        const int hidden = get_i(32);
        const std::string load_txt = get_s("best_genome.txt");

        AleEvaluator evaluator(rom, /*seed=*/123, /*render=*/true, steps);
        const int nActions = (int)evaluator.actions().size();

        int file_hidden = 0, file_actions = 0;
        std::vector<double> genome = load_genome_txt(load_txt, file_hidden, file_actions);

        if (file_hidden != hidden) {
            std::cerr << "[WARN] hidden del fichero (" << file_hidden
                      << ") != hidden pasado (" << hidden << ")\n";
        }
        if (file_actions != nActions) {
            std::cerr << "[WARN] actions del fichero (" << file_actions
                      << ") != actions del juego (" << nActions << ")\n";
        }

        std::cout << "PLAY: usando pesos desde: " << load_txt << "\n";
        (void)evaluator.rollout(genome, hidden, /*episodes=*/1);
        return 0;
    }

    // ---------------- TRAIN ----------------
    const int gens     = get_i(200);
    const int pop      = get_i(100);
    const int steps    = get_i(500);
    const int episodes = get_i(1);
    const int hidden   = get_i(32);
    const std::string save_header = get_s("best_genome.hpp");
    const std::string save_txt    = get_s("best_genome.txt");

    AleEvaluator evaluator(rom, /*seed=*/123, /*render=*/false, steps);
    const int nActions = (int)evaluator.actions().size();

    const std::size_t dim = MLPPolicy::genome_size(RAM_BYTES, hidden, nActions);

    rcga::RealCodedGA::Config cfg;
    cfg.dim = dim;
    cfg.population_size = (std::size_t)pop;
    cfg.generations = (std::size_t)gens;
    cfg.objective = rcga::RealCodedGA::Objective::Maximize;

    // Operadores (puedes tocar esto)
    cfg.selection = rcga::RealCodedGA::Selection::Tournament;
    cfg.tournament_k = 3;
    cfg.crossover = rcga::RealCodedGA::Crossover::SBX;
    cfg.crossover_prob = 0.9;
    cfg.mutation = rcga::RealCodedGA::Mutation::Polynomial;
    cfg.mutation_prob = -1.0; // 1/dim

    // Bounds de pesos
    cfg.uniform_bounds.lo = -2.0;
    cfg.uniform_bounds.hi =  2.0;

    // seed=0 => random_device
    cfg.seed = 0;

    rcga::RealCodedGA ga(cfg);

    ga.set_fitness([&](const rcga::RealCodedGA::Vec& genome) -> double {
        return evaluator.rollout(genome, hidden, episodes);
    });

    ga.set_progress_callback([&](std::size_t gen, double best, double mean) {
        std::cout << "gen=" << gen << " best=" << best << " mean=" << mean << "\n";
    });

    auto res = ga.run();

    std::cout << "\n=== DONE ===\n";
    std::cout << "best_fitness=" << res.best_fitness
              << " at gen=" << res.best_generation
              << " dim=" << dim
              << " hidden=" << hidden
              << " actions=" << nActions << "\n";

    // Guardar (txt para play sin recompilar, hpp para include con recompilación)
    save_genome_header(save_header, hidden, nActions, res.best_x);
    save_genome_txt(save_txt, hidden, nActions, res.best_x);

    std::cout << "Saved header: " << save_header << "\n";
    std::cout << "Saved txt:    " << save_txt << "\n";

    // Demo renderizada con el mejor
    std::cout << "\nRunning demo with display...\n";
    AleEvaluator demo(rom, /*seed=*/123, /*render=*/true, steps);
    (void)demo.rollout(res.best_x, hidden, /*episodes=*/1);

    return 0;
}
