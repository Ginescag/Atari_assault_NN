// rcga_train_ale.cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <array>

#include "src/ale_interface.hpp" // igual que minimal_agent.cpp
#include "ga.cpp"                // TU RealCodedGA (sí: se incluye como "header")

// -------------------- RAM features (inputs del MLP) --------------------
// Nota: NO metemos enemigos todavía. Solo movimiento + disparo/cooldown.
static constexpr std::array<int, 9> FEAT = {
    0x7E, // X fina / movimiento
    0x5C, // X gruesa / cuantizada
    0x7F, // flags/estado ligado al movimiento
    0x7C, // señal secundaria izq/der
    0x7D, // señal secundaria
    0x5D, // secundaria
    0x6A, // disparo/proyectil activo
    0x3E, // estado/cooldown disparo
    0x43  // timer/cooldown disparo
};
static constexpr int IN_DIM = (int)FEAT.size();

// -------------------- Training knobs --------------------
static constexpr int    TRAIN_NOOPS_MAX           = 30;    // 0..30 NOOPs al inicio
static constexpr double TRAIN_EPSILON             = 0.00;  // GA ya explora: mejor 0
static constexpr int    REPEAT_PENALTY_STREAK     = 25;
static constexpr double REPEAT_PENALTY_PER_STEP   = 0.05;

// -------------------- Fitness shaping (anti "camp center + fire") --------------------
// IMPORTANTE: ahora la presión viene del SCORE real (reward), no de "vivir/centro".
static constexpr double W_SCORE = 50.0;  // sube mucho el peso del reward real
static constexpr double W_ALIVE = 0.0;   // quita premio por vivir (evita "camping")

// Quieto / movimiento (usando delta de 0x7E)
static constexpr int    PLAYER_X_IDX     = 0x7E;
static constexpr int    STILL_STREAK     = 40;
static constexpr double STILL_PENALTY    = 0.05;

// Centro / borde
static constexpr int    EDGE_MARGIN      = 20;
static constexpr int    EDGE_STREAK      = 60;
static constexpr double EDGE_PENALTY     = 0.01;
static constexpr double CENTER_BONUS     = 0.0;  // DESACTIVADO: fomentaba "me quedo en el centro"

// Anti-spam disparo: castiga disparar mucho rato sin obtener reward (independiente de RAM)
static constexpr int    FIRE_NOREWARD_STREAK  = 60;
static constexpr double FIRE_NOREWARD_PENALTY = 0.03;

// Stall robusto: si los FEAT apenas cambian durante mucho rato, está “atascado”
static constexpr int    STALL_STREAK      = 120;
static constexpr double STALL_PENALTY     = 0.05;

// Anti-spam extra basado en tus bytes (opcional, lo dejo suave)
static constexpr int    BULLET_IDX       = 0x6A;
static constexpr int    COOLDOWN_A       = 0x3E;
static constexpr int    COOLDOWN_B       = 0x43;
static constexpr double BAD_FIRE_PENALTY = 0.01;

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

// -------------------- MLP policy (FEAT RAM -> logits -> argmax action) --------------------

struct MLPPolicy {
    int in = IN_DIM;
    int hidden = 8;
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

    // Normaliza byte -> [-1, 1]
    static inline double norm_byte(uint8_t v) {
        return (double(v) / 255.0) * 2.0 - 1.0;
    }

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
                const uint8_t raw = (uint8_t)ram.get(FEAT[(std::size_t)i]);
                const double xi = norm_byte(raw);
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
                 int maxSteps,
                 bool training,
                 float repeat_action_prob)
    : rom_(rom), seed_(seed), render_(render), maxSteps_(maxSteps), training_(training) {

        ale_.setInt("random_seed", seed_);
        ale_.setFloat("repeat_action_probability", repeat_action_prob);
        ale_.setBool("display_screen", render_);
        ale_.setBool("sound", render_);

        ale_.loadROM(rom_);

        actions_ = ale_.getMinimalActionSet();
        if (actions_.empty()) {
            throw std::runtime_error("ALE: minimal action set empty");
        }

        ale_.reset_game();
        startState_ = ale_.cloneSystemState();

        hasFire_ = (std::find(actions_.begin(), actions_.end(), PLAYER_A_FIRE) != actions_.end());
    }

    const std::vector<Action>& actions() const { return actions_; }

    static inline bool is_fire_action(Action a) {
        return a == PLAYER_A_FIRE || a == PLAYER_A_LEFTFIRE || a == PLAYER_A_RIGHTFIRE;
    }

    // dx con wrap 0..255 -> [-128,127]
    static inline int wrap_delta(uint8_t cur, uint8_t prev) {
        int d = (int)cur - (int)prev;
        if (d > 127) d -= 256;
        if (d < -127) d += 256;
        return d;
    }

    double rollout(const std::vector<double>& genome, int hidden, int episodes) {
        MLPPolicy pol;
        pol.bind(genome, hidden, (int)actions_.size());

        double total = 0.0;

        for (int ep = 0; ep < episodes; ++ep) {
            std::mt19937 rng((uint32_t)(seed_ * 10007u + (unsigned)ep * 1337u));
            std::uniform_int_distribution<int> noops_d(0, TRAIN_NOOPS_MAX);
            std::uniform_real_distribution<double> ur(0.0, 1.0);
            std::uniform_int_distribution<int> ua(0, (int)actions_.size() - 1);

            if (training_) {
                ale_.reset_game();

                // NOOPs aleatorios al inicio para variar estado
                const int noops = noops_d(rng);
                for (int i = 0; i < noops && !ale_.game_over(); ++i) {
                    (void)ale_.act(PLAYER_A_NOOP);
                }
            } else {
                ale_.restoreSystemState(startState_);
            }

            // Arranque típico
            if (hasFire_) {
                for (int i = 0; i < 3; ++i) ale_.act(PLAYER_A_FIRE);
            }

            double epFit = 0.0;

            int prev_k = -1;
            int streak = 0;

            int still = 0;
            int edge = 0;
            bool hasPrevX = false;
            uint8_t prevX = 0;

            // NUEVO: stall robusto por cambios en FEAT + “fire sin reward”
            std::array<uint8_t, FEAT.size()> prevFeat{};
            bool hasPrevFeat = false;
            int stall = 0;
            int fireNoReward = 0;

            int step = 0;
            while (!ale_.game_over() && step < maxSteps_) {
                auto ram = ale_.getRAM();

                // --- Stall robusto: cuántos FEAT cambian entre frames ---
                std::array<uint8_t, FEAT.size()> curFeat{};
                for (size_t i = 0; i < FEAT.size(); ++i) curFeat[i] = (uint8_t)ram.get(FEAT[i]);

                if (hasPrevFeat) {
                    int changed = 0;
                    for (size_t i = 0; i < FEAT.size(); ++i) {
                        if (curFeat[i] != prevFeat[i]) ++changed;
                    }
                    if (changed <= 1) ++stall;   // casi no cambia nada -> atascado
                    else stall = 0;
                } else {
                    hasPrevFeat = true;
                    stall = 0;
                }
                prevFeat = curFeat;

                // --- señales de movimiento / posición ---
                const uint8_t x = (uint8_t)ram.get(PLAYER_X_IDX);

                int dx = 0;
                if (hasPrevX) dx = wrap_delta(x, prevX);
                hasPrevX = true;
                prevX = x;

                if (std::abs(dx) < 2) ++still;
                else still = 0;

                const bool onEdge = (x < EDGE_MARGIN) || (x > (255 - EDGE_MARGIN));
                if (onEdge) ++edge;
                else edge = 0;

                // anti-spam extra por RAM (suave)
                const bool bulletActive = ((uint8_t)ram.get(BULLET_IDX)) != 0;
                const bool cooldown = ((uint8_t)ram.get(COOLDOWN_A) >= 30) || ((uint8_t)ram.get(COOLDOWN_B) != 127);

                // --- elegir acción ---
                int k = pol.act_index(ram);
                if (k < 0) k = 0;
                if (k >= (int)actions_.size()) k = (int)actions_.size() - 1;

                if (training_ && TRAIN_EPSILON > 0.0 && ur(rng) < TRAIN_EPSILON) {
                    k = ua(rng);
                }

                if (training_) {
                    if (k == prev_k) ++streak;
                    else { prev_k = k; streak = 0; }
                }

                const Action a = actions_[(std::size_t)k];
                reward_t r = ale_.act(a);

                // --- Fitness base: SOLO score real (y opcional alive) ---
                epFit += W_SCORE * (double)r;
                epFit += W_ALIVE;

                // --- Shaping SOLO training ---
                if (training_) {
                    // repetir acción
                    if (streak > REPEAT_PENALTY_STREAK) epFit -= REPEAT_PENALTY_PER_STEP;

                    // quieto demasiado
                    if (still > STILL_STREAK) epFit -= STILL_PENALTY;

                    // borde demasiado
                    if (edge > EDGE_STREAK) epFit -= EDGE_PENALTY;

                    // stall (FEAT casi no cambia)
                    if (stall > STALL_STREAK) epFit -= STALL_PENALTY;

                    // NUEVO: penaliza disparar mucho rato sin reward
                    const bool fire = is_fire_action(a);
                    if (fire && r == 0) ++fireNoReward;
                    else fireNoReward = 0;

                    if (fireNoReward > FIRE_NOREWARD_STREAK) {
                        epFit -= FIRE_NOREWARD_PENALTY;
                    }

                    // anti-spam extra: disparar cuando ya hay bala/cooldown y no da puntos
                    if (fire && (bulletActive || cooldown) && r == 0) {
                        epFit -= BAD_FIRE_PENALTY;
                    }
                }

                ++step;
            }

            total += epFit;
        }

        return total / std::max(1, episodes);
    }

private:
    std::string rom_;
    int seed_ = 0;
    bool render_ = false;
    int maxSteps_ = 500;
    bool training_ = false;

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
          << "  " << argv[0] << " <romfile> train [gens=400] [pop=150] [steps=20000] [episodes=5] [hidden=8]\n"
          << "       [save_header=best_genome.hpp] [save_txt=best_genome.txt]\n"
          << "  " << argv[0] << " <romfile> play  [steps=20000] [hidden=8] [load_txt=best_genome.txt]\n";
        return 1;
    }

    const std::string rom = argv[1];
    std::string mode = "train";
    int argi = 2;

    if (argc >= 3) {
        std::string m = argv[2];
        if (m == "train" || m == "play" || m == "display") {
            mode = (m == "display") ? "play" : m;
            argi = 3;
        }
    }

    auto get_i = [&](int def) -> int {
        if (argi < argc) {
            const std::string s = std::string(argv[argi++]);
            return std::stoi(s);
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
        const int hidden = get_i(8);
        const std::string load_txt = get_s("best_genome.txt");

        AleEvaluator evaluator(rom, /*seed=*/123, /*render=*/true, steps,
                               /*training=*/false,
                               /*repeat_action_prob=*/0.25f); // coherente con train
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
    const int gens     = get_i(400);
    const int pop      = get_i(150);
    const int steps    = get_i(20000);
    const int episodes = get_i(5);
    const int hidden   = get_i(8);
    const std::string save_header = get_s("best_genome.hpp");
    const std::string save_txt    = get_s("best_genome.txt");

    AleEvaluator evaluator(rom, /*seed=*/123, /*render=*/false, steps,
                           /*training=*/true,
                           /*repeat_action_prob=*/0.25f); // coherente con play/demo
    const int nActions = (int)evaluator.actions().size();

    const std::size_t dim = MLPPolicy::genome_size(IN_DIM, hidden, nActions);

    rcga::RealCodedGA::Config cfg;
    cfg.dim = dim;
    cfg.population_size = (std::size_t)pop;
    cfg.generations = (std::size_t)gens;
    cfg.objective = rcga::RealCodedGA::Objective::Maximize;

    cfg.selection = rcga::RealCodedGA::Selection::Tournament;
    cfg.tournament_k = 3;
    cfg.crossover = rcga::RealCodedGA::Crossover::SBX;
    cfg.crossover_prob = 0.9;
    cfg.mutation = rcga::RealCodedGA::Mutation::Polynomial;
    cfg.mutation_prob = -2.0; // 1/dim (según tu implementación)

    cfg.uniform_bounds.lo = -2.0;
    cfg.uniform_bounds.hi =  2.0;
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

    save_genome_header(save_header, hidden, nActions, res.best_x);
    save_genome_txt(save_txt, hidden, nActions, res.best_x);

    std::cout << "Saved header: " << save_header << "\n";
    std::cout << "Saved txt:    " << save_txt << "\n";

    std::cout << "\nRunning demo with display...\n";
    AleEvaluator demo(rom, /*seed=*/123, /*render=*/true, steps,
                      /*training=*/false,
                      /*repeat_action_prob=*/0.25f);
    (void)demo.rollout(res.best_x, hidden, /*episodes=*/1);

    return 0;
}
