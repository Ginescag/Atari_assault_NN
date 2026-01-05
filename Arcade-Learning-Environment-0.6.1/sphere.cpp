#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <cstdint>
#include "ga.cpp" 

static double sphere(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += v * v;
    return s;
}

static bool within_bounds(const std::vector<double>& x,
                          const std::vector<rcga::Bounds>& b) {
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] < b[i].lo - 1e-12 || x[i] > b[i].hi + 1e-12) return false;
    }
    return true;
}

static void print_vector(const std::vector<double>& x, size_t maxn = 8) {
    std::cout << "[";
    for (size_t i = 0; i < x.size() && i < maxn; ++i) {
        std::cout << (i ? ", " : "") << x[i];
    }
    if (x.size() > maxn) std::cout << ", ...";
    std::cout << "]";
}

int main() {
    using namespace rcga;

    // -------------------------
    // TEST 1: MINIMIZE Sphere
    // -------------------------
    {
        RealCodedGA ga;
        RealCodedGA::Config cfg;

        cfg.dim = 10;
        cfg.bounds.assign(cfg.dim, Bounds{-5.0, 5.0});

        cfg.population_size = 200;
        cfg.generations = 250;

        cfg.objective = RealCodedGA::Objective::Minimize;

        cfg.selection = RealCodedGA::Selection::Tournament;
        cfg.tournament_k = 3;

        cfg.crossover = RealCodedGA::Crossover::SBX;
        cfg.crossover_prob = 0.9;
        cfg.sbx_eta = 15.0;

        cfg.mutation = RealCodedGA::Mutation::Polynomial;
        cfg.mutation_prob = -1.0;   // default 1/dim
        cfg.poly_eta = 20.0;

        cfg.elite_count = 1;

        cfg.seed = 92345; // IMPORTANT: reproducible

        ga.set_config(cfg);
        ga.set_fitness([](const std::vector<double>& x) {
            return sphere(x);
        });

        ga.set_progress_callback([](size_t gen, double best, double mean) {
            if (gen % 25 == 0) {
                std::cout << "MIN gen " << std::setw(3) << gen
                          << " | best=" << std::scientific << best
                          << " | mean=" << mean << "\n";
            }
        });

        auto res = ga.run();

        std::cout << "\n[TEST 1 - MIN Sphere]\n";
        std::cout << "Best fitness: " << std::scientific << res.best_fitness
                  << " at gen " << res.best_generation << "\n";
        std::cout << "Best x: ";
        print_vector(res.best_x);
        std::cout << "\n";

        bool ok_bounds = within_bounds(res.best_x, cfg.bounds);
        std::cout << "Bounds OK: " << (ok_bounds ? "YES" : "NO") << "\n";

        // Umbral realista para GA (no uses 1e-12 aquí: es estocástico)
        const double threshold = 1e-4;
        bool ok = std::isfinite(res.best_fitness) && (res.best_fitness < threshold) && ok_bounds;

        std::cout << "PASS criteria (best < " << threshold << "): "
                  << (ok ? "PASS" : "FAIL") << "\n\n";
    }

    // -------------------------------------
    // TEST 2: MAXIMIZE (-Sphere) -> debe ir a 0
    // -------------------------------------
    {
        RealCodedGA ga;
        RealCodedGA::Config cfg;

        cfg.dim = 10;
        cfg.bounds.assign(cfg.dim, Bounds{-5.0, 5.0});

        cfg.population_size = 200;
        cfg.generations = 250;

        cfg.objective = RealCodedGA::Objective::Maximize;

        cfg.selection = RealCodedGA::Selection::Tournament;
        cfg.tournament_k = 3;

        cfg.crossover = RealCodedGA::Crossover::SBX;
        cfg.crossover_prob = 0.9;

        cfg.mutation = RealCodedGA::Mutation::Polynomial;
        cfg.mutation_prob = -1.0;

        cfg.elite_count = 1;
        cfg.seed = 123456; // misma semilla => comportamiento comparable

        // objetivo: como maximizamos, el mejor valor debería acercarse a 0 (desde negativo)
        cfg.use_target_fitness = true;
        cfg.target_fitness = -1e-4; // cuando best >= -1e-4, parar

        ga.set_config(cfg);
        ga.set_fitness([](const std::vector<double>& x) {
            return -sphere(x);
        });

        ga.set_progress_callback([](size_t gen, double best, double mean) {
            if (gen % 25 == 0) {
                std::cout << "MAX gen " << std::setw(3) << gen
                          << " | best=" << std::scientific << best
                          << " | mean=" << mean << "\n";
            }
        });

        auto res = ga.run();

        std::cout << "\n[TEST 2 - MAX (-Sphere)]\n";
        std::cout << "Best fitness: " << std::scientific << res.best_fitness
                  << " at gen " << res.best_generation << "\n";
        std::cout << "Best x: ";
        print_vector(res.best_x);
        std::cout << "\n";

        bool ok_bounds = within_bounds(res.best_x, cfg.bounds);
        std::cout << "Bounds OK: " << (ok_bounds ? "YES" : "NO") << "\n";

        // En maximización, queremos best cercano a 0 y >= -threshold
        const double threshold = 1e-4;
        bool ok = std::isfinite(res.best_fitness) && (res.best_fitness >= -threshold) && ok_bounds;

        std::cout << "PASS criteria (best >= " << -threshold << "): "
                  << (ok ? "PASS" : "FAIL") << "\n\n";
    }

    std::cout << "Done.\n";
    return 0;
}
