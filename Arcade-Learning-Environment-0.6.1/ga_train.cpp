#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>
#include <cstdint>
#include <cmath>

#include "src/ale_interface.hpp"
#include "NeuralNetwork.h"

using std::string;

static constexpr int INPUTS  = 128;
static constexpr int HIDDEN  = 32;
static constexpr int OUTPUTS = 7;

static const Action ACTIONS[OUTPUTS] = {
    PLAYER_A_NOOP,
    PLAYER_A_FIRE,
    PLAYER_A_LEFT,
    PLAYER_A_RIGHT,
    PLAYER_A_LEFTFIRE,
    PLAYER_A_RIGHTFIRE,
    PLAYER_A_UPFIRE
};

struct GAConfig {
    int population = 50;
    int generations = 40;
    int elite = 2;

    int episodes_per_individual = 2;   // train fast; validate later with 3
    int max_steps = 6000;             // safety cap
    int base_seed = 0;

    int tournament_k = 3;

    double mutation_prob = 0.10;       // per gene
    double mutation_sigma = 0.08;      // gaussian std
};

static inline double normalize_ram(uint8_t b) {
    return (double(b) / 127.5) - 1.0; // [0,255] -> [-1,1]
}

static int argmax_colvec(const Matrix& out) {
    // out is (OUTPUTS x 1)
    int best = 0;
    double bestv = out.at(0,0);
    for (unsigned i = 1; i < out.getRows(); ++i) {
        double v = out.at(i,0);
        if (v > bestv) { bestv = v; best = (int)i; }
    }
    return best;
}

static double run_episode(ALEInterface& ale, NeuralNetwork& nn, int max_steps) {
    ale.reset_game();
    double total = 0.0;

    int lives = ale.lives();
    int step = 0;

    while (!ale.game_over() && step < max_steps) {
        // same trick you already use: fire when life decreases
        if (ale.lives() < lives) {
            lives = ale.lives();
            ale.act(PLAYER_A_FIRE);
        }

        auto ram = ale.getRAM();
        Matrix x(INPUTS, 1);
        for (int i = 0; i < INPUTS; ++i) {
            x.at(i,0) = normalize_ram((uint8_t)ram.get(i));
        }

        Matrix out = nn.feedforward(x, "tanh");
        int a = argmax_colvec(out);

        reward_t r = ale.act(ACTIONS[a]);
        total += (double)r;

        ++step;
    }
    return total;
}

static double eval_fitness(const std::vector<double>& genome,
                           const string& rom_path,
                           const GAConfig& cfg,
                           int seed_offset)
{
    ALEInterface ale;
    ale.setFloat("repeat_action_probability", 0.0);
    ale.setInt("random_seed", cfg.base_seed + seed_offset);

    // For training speed:
    ale.setBool("display_screen", false);
    ale.setBool("sound", false);

    ale.loadROM(rom_path);

    NeuralNetwork nn({INPUTS, HIDDEN, OUTPUTS});
    nn.setParams(genome);

    double sum = 0.0;
    for (int ep = 0; ep < cfg.episodes_per_individual; ++ep) {
        ale.setInt("random_seed", cfg.base_seed + seed_offset + 1000 * ep);
        sum += run_episode(ale, nn, cfg.max_steps);
    }
    return sum / (double)cfg.episodes_per_individual;
}

static int tournament_select(const std::vector<double>& fit,
                             int k,
                             std::mt19937& rng)
{
    std::uniform_int_distribution<int> dist(0, (int)fit.size()-1);
    int best = dist(rng);
    for (int i = 1; i < k; ++i) {
        int j = dist(rng);
        if (fit[j] > fit[best]) best = j;
    }
    return best;
}

static std::vector<double> crossover_uniform(const std::vector<double>& a,
                                             const std::vector<double>& b,
                                             std::mt19937& rng)
{
    std::bernoulli_distribution pick(0.5);
    std::vector<double> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = pick(rng) ? a[i] : b[i];
    return c;
}

static void mutate_gaussian(std::vector<double>& g,
                            double p,
                            double sigma,
                            std::mt19937& rng)
{
    std::bernoulli_distribution do_mut(p);
    std::normal_distribution<double> noise(0.0, sigma);
    for (double& w : g) {
        if (do_mut(rng)) w += noise(rng);
    }
}

static void write_header(const string& path,
                         const std::vector<double>& best)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write header: " + path);

    f << "#pragma once\n";
    f << "#include <cstddef>\n";
    f << "static constexpr std::size_t GA_PARAM_COUNT = " << best.size() << ";\n";
    f << "static const double GA_PARAMS[GA_PARAM_COUNT] = {\n";
    for (size_t i = 0; i < best.size(); ++i) {
        f << "  " << best[i];
        f << (i + 1 == best.size() ? "\n" : ",\n");
    }
    f << "};\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./ga_train supported/assault.bin [out_header=ga_weights.h]\n";
        return 1;
    }
    string rom_path = argv[1];
    string out_header = (argc >= 3) ? argv[2] : "ga_weights.h";

    GAConfig cfg;

    NeuralNetwork proto({INPUTS, HIDDEN, OUTPUTS});
    const size_t G = proto.paramCount();

    std::mt19937 rng(12345);
    std::normal_distribution<double> init(0.0, 0.5);

    std::vector<std::vector<double>> pop(cfg.population, std::vector<double>(G));
    for (auto& ind : pop) for (double& w : ind) w = init(rng);

    std::vector<double> fit(cfg.population, -1e18);
    std::vector<double> best_genome;
    double best_fit = -1e18;

    for (int gen = 0; gen < cfg.generations; ++gen) {
        // evaluate
        for (int i = 0; i < cfg.population; ++i) {
            fit[i] = eval_fitness(pop[i], rom_path, cfg, gen * 10000 + i);
        }

        int bi = (int)std::distance(fit.begin(), std::max_element(fit.begin(), fit.end()));
        double gen_best = fit[bi];
        double gen_avg = std::accumulate(fit.begin(), fit.end(), 0.0) / fit.size();

        if (gen_best > best_fit) {
            best_fit = gen_best;
            best_genome = pop[bi];
        }

        std::cout << "[GEN " << gen << "] best=" << gen_best
                  << " avg=" << gen_avg
                  << " global_best=" << best_fit << "\n";

        // sort indices by fitness (desc)
        std::vector<int> idx(cfg.population);
        for (int i = 0; i < cfg.population; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return fit[a] > fit[b]; });

        // next generation
        std::vector<std::vector<double>> next;
        next.reserve(cfg.population);

        // elites
        for (int e = 0; e < cfg.elite; ++e) next.push_back(pop[idx[e]]);

        while ((int)next.size() < cfg.population) {
            int p1 = tournament_select(fit, cfg.tournament_k, rng);
            int p2 = tournament_select(fit, cfg.tournament_k, rng);
            auto child = crossover_uniform(pop[p1], pop[p2], rng);
            mutate_gaussian(child, cfg.mutation_prob, cfg.mutation_sigma, rng);
            next.push_back(std::move(child));
        }

        pop = std::move(next);
    }

    write_header(out_header, best_genome);
    std::cout << "Wrote best genome to: " << out_header
              << " (fitness=" << best_fit << ")\n";
    return 0;
}
