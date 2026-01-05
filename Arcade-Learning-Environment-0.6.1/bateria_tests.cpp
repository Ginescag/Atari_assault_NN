#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>

#include "ga.cpp" 

// ----------------------------
// Utilidades
// ----------------------------
static bool g_verbose = false;

static double sphere(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += v * v;
    return s;
}

static bool within_bounds(const std::vector<double>& x,
                          const std::vector<rcga::Bounds>& b,
                          double eps = 1e-12) {
    if (x.size() != b.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] < b[i].lo - eps || x[i] > b[i].hi + eps) return false;
    }
    return true;
}

static bool is_finite(double v) { return std::isfinite(v) != 0; }

static bool approx_ge(double a, double b, double eps = 1e-12) {
    return (a > b) || std::fabs(a - b) <= eps;
}

static std::string vec_head(const std::vector<double>& x, size_t maxn = 8) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < x.size() && i < maxn; ++i) {
        oss << (i ? ", " : "") << std::scientific << x[i];
    }
    if (x.size() > maxn) oss << ", ...";
    oss << "]";
    return oss.str();
}

struct TestCaseResult {
    std::string name;
    bool pass = false;
    std::string details;
};

static TestCaseResult make_fail(std::string name, std::string details) {
    return TestCaseResult{std::move(name), false, std::move(details)};
}
static TestCaseResult make_pass(std::string name, std::string details = {}) {
    return TestCaseResult{std::move(name), true, std::move(details)};
}

static rcga::RealCodedGA::Config base_cfg(size_t dim,
                                         std::vector<rcga::Bounds> bounds,
                                         std::uint64_t seed) {
    using GA = rcga::RealCodedGA;
    GA::Config cfg;
    cfg.dim = dim;
    cfg.bounds = std::move(bounds);

    cfg.population_size = 200;
    cfg.generations = 250;

    cfg.selection = GA::Selection::Tournament;
    cfg.tournament_k = 3;

    cfg.crossover = GA::Crossover::SBX;
    cfg.crossover_prob = 0.9;
    cfg.sbx_eta = 15.0;

    cfg.mutation = GA::Mutation::Polynomial;
    cfg.mutation_prob = -1.0; // default 1/dim
    cfg.poly_eta = 20.0;

    cfg.elite_count = 1;
    cfg.seed = seed;
    return cfg;
}

// ----------------------------
// TESTS
// ----------------------------
static TestCaseResult run_sphere_min_tournament() {
    using GA = rcga::RealCodedGA;

    const std::string name = "Sphere MIN (Tournament + SBX + Poly)";
    const size_t dim = 10;
    std::vector<rcga::Bounds> b(dim, rcga::Bounds{-5.0, 5.0});

    GA ga;
    auto cfg = base_cfg(dim, b, 123456);
    cfg.objective = GA::Objective::Minimize;

    ga.set_config(cfg);
    ga.set_fitness([](const std::vector<double>& x) { return sphere(x); });

    if (g_verbose) {
        ga.set_progress_callback([](size_t gen, double best, double mean) {
            if (gen % 25 == 0) {
                std::cout << "  gen " << std::setw(3) << gen
                          << " | best=" << std::scientific << best
                          << " | mean=" << mean << "\n";
            }
        });
    }

    auto res = ga.run();

    if (res.best_x.size() != dim) return make_fail(name, "best_x dim mismatch.");
    if (!is_finite(res.best_fitness)) return make_fail(name, "best_fitness is not finite.");
    if (!within_bounds(res.best_x, cfg.bounds)) return make_fail(name, "best_x violates bounds.");
    if (res.history_best.empty()) return make_fail(name, "history_best is empty.");

    const double threshold = 1e-4;
    if (!(res.best_fitness < threshold)) {
        std::ostringstream oss;
        oss << "Expected best_fitness < " << std::scientific << threshold
            << " but got " << res.best_fitness
            << " | best_x=" << vec_head(res.best_x);
        return make_fail(name, oss.str());
    }

    std::ostringstream ok;
    ok << "best=" << std::scientific << res.best_fitness
       << " gen=" << res.best_generation
       << " x=" << vec_head(res.best_x);
    return make_pass(name, ok.str());
}

static TestCaseResult run_sphere_max_neg_sphere_target_stop() {
    using GA = rcga::RealCodedGA;

    const std::string name = "MAX (-Sphere) with target stop";
    const size_t dim = 10;
    std::vector<rcga::Bounds> b(dim, rcga::Bounds{-5.0, 5.0});

    GA ga;
    auto cfg = base_cfg(dim, b, 123456);
    cfg.objective = GA::Objective::Maximize;

    cfg.use_target_fitness = true;
    cfg.target_fitness = -1e-4;

    ga.set_config(cfg);
    ga.set_fitness([](const std::vector<double>& x) { return -sphere(x); });

    auto res = ga.run();

    if (res.best_x.size() != dim) return make_fail(name, "best_x dim mismatch.");
    if (!is_finite(res.best_fitness)) return make_fail(name, "best_fitness is not finite.");
    if (!within_bounds(res.best_x, cfg.bounds)) return make_fail(name, "best_x violates bounds.");

    const double threshold = 1e-4;
    if (!approx_ge(res.best_fitness, -threshold)) {
        std::ostringstream oss;
        oss << "Expected best_fitness >= " << std::scientific << (-threshold)
            << " but got " << res.best_fitness
            << " | best_x=" << vec_head(res.best_x);
        return make_fail(name, oss.str());
    }

    std::ostringstream ok;
    ok << "best=" << std::scientific << res.best_fitness
       << " gen=" << res.best_generation
       << " x=" << vec_head(res.best_x);
    return make_pass(name, ok.str());
}

static TestCaseResult run_dim1_edge_case() {
    using GA = rcga::RealCodedGA;

    const std::string name = "Edge case dim=1 (Sphere MIN)";
    const size_t dim = 1;
    std::vector<rcga::Bounds> b(dim, rcga::Bounds{-5.0, 5.0});

    GA ga;
    auto cfg = base_cfg(dim, b, 777);
    cfg.objective = GA::Objective::Minimize;

    cfg.population_size = 120;
    cfg.generations = 150;

    ga.set_config(cfg);
    ga.set_fitness([](const std::vector<double>& x) { return sphere(x); });

    auto res = ga.run();

    if (res.best_x.size() != dim) return make_fail(name, "best_x dim mismatch.");
    if (!is_finite(res.best_fitness)) return make_fail(name, "best_fitness not finite.");
    if (!within_bounds(res.best_x, cfg.bounds)) return make_fail(name, "bounds violation.");

    const double threshold = 1e-6;
    if (!(res.best_fitness < threshold)) {
        std::ostringstream oss;
        oss << "Expected best_fitness < " << std::scientific << threshold
            << " but got " << res.best_fitness
            << " | x=" << vec_head(res.best_x);
        return make_fail(name, oss.str());
    }

    std::ostringstream ok;
    ok << "best=" << std::scientific << res.best_fitness
       << " x=" << vec_head(res.best_x);
    return make_pass(name, ok.str());
}

static TestCaseResult run_per_gene_bounds() {
    using GA = rcga::RealCodedGA;

    const std::string name = "Per-gene bounds (Sphere MIN)";
    const size_t dim = 5;
    std::vector<rcga::Bounds> b = {
        {-1.0, 1.0},
        {-5.0, 5.0},
        {-0.1, 0.1},
        {-10.0, 10.0},
        {-2.0, 3.0}
    };

    GA ga;
    auto cfg = base_cfg(dim, b, 999);
    cfg.objective = GA::Objective::Minimize;
    cfg.population_size = 180;
    cfg.generations = 220;

    ga.set_config(cfg);
    ga.set_fitness([](const std::vector<double>& x) { return sphere(x); });

    auto res = ga.run();

    if (res.best_x.size() != dim) return make_fail(name, "best_x dim mismatch.");
    if (!is_finite(res.best_fitness)) return make_fail(name, "best_fitness not finite.");
    if (!within_bounds(res.best_x, cfg.bounds)) return make_fail(name, "bounds violation.");

    const double threshold = 1e-6;
    if (!(res.best_fitness < threshold)) {
        std::ostringstream oss;
        oss << "Expected best_fitness < " << std::scientific << threshold
            << " but got " << res.best_fitness
            << " | x=" << vec_head(res.best_x);
        return make_fail(name, oss.str());
    }

    std::ostringstream ok;
    ok << "best=" << std::scientific << res.best_fitness
       << " x=" << vec_head(res.best_x);
    return make_pass(name, ok.str());
}

static TestCaseResult run_nan_zone_handling() {
    using GA = rcga::RealCodedGA;

    const std::string name = "NaN-zone handling (x[0] > 4 => NaN)";
    const size_t dim = 10;
    std::vector<rcga::Bounds> b(dim, rcga::Bounds{-5.0, 5.0});

    GA ga;
    auto cfg = base_cfg(dim, b, 424242);
    cfg.objective = GA::Objective::Minimize;
    cfg.generations = 300;

    ga.set_config(cfg);
    ga.set_fitness([](const std::vector<double>& x) {
        if (x[0] > 4.0) return std::numeric_limits<double>::quiet_NaN();
        return sphere(x);
    });

    auto res = ga.run();

    if (res.best_x.size() != dim) return make_fail(name, "best_x dim mismatch.");
    if (!is_finite(res.best_fitness)) return make_fail(name, "best_fitness not finite.");
    if (!within_bounds(res.best_x, cfg.bounds)) return make_fail(name, "bounds violation.");

    if (!(res.best_x[0] <= 4.0 + 1e-12)) {
        std::ostringstream oss;
        oss << "Best solution is inside NaN zone (x0=" << res.best_x[0] << ").";
        return make_fail(name, oss.str());
    }

    const double threshold = 1e-4;
    if (!(res.best_fitness < threshold)) {
        std::ostringstream oss;
        oss << "Expected best_fitness < " << std::scientific << threshold
            << " but got " << res.best_fitness
            << " | x=" << vec_head(res.best_x);
        return make_fail(name, oss.str());
    }

    std::ostringstream ok;
    ok << "best=" << std::scientific << res.best_fitness
       << " x0=" << std::scientific << res.best_x[0]
       << " x=" << vec_head(res.best_x);
    return make_pass(name, ok.str());
}

// ✅ AJUSTE 1: Roulette SIN Gaussiana (ruido constante) para no falsear el test.
// Mantiene la ruta Roulette, pero permite “afinar” cerca del óptimo.
static TestCaseResult run_roulette_selection() {
    using GA = rcga::RealCodedGA;

    const std::string name = "Roulette selection (Sphere MIN, Poly)";
    const size_t dim = 10;
    std::vector<rcga::Bounds> b(dim, rcga::Bounds{-5.0, 5.0});

    GA ga;
    auto cfg = base_cfg(dim, b, 20240101);
    cfg.objective = GA::Objective::Minimize;

    cfg.selection = GA::Selection::Roulette;

    // Un pelín más de tiempo, porque Roulette suele tener menos “presión”
    cfg.population_size = 260;
    cfg.generations = 400;

    ga.set_config(cfg);
    ga.set_fitness([](const std::vector<double>& x) { return sphere(x); });

    auto res = ga.run();

    if (res.best_x.size() != dim) return make_fail(name, "best_x dim mismatch.");
    if (!is_finite(res.best_fitness)) return make_fail(name, "best_fitness not finite.");
    if (!within_bounds(res.best_x, cfg.bounds)) return make_fail(name, "bounds violation.");

    const double threshold = 5e-4;
    if (!(res.best_fitness < threshold)) {
        std::ostringstream oss;
        oss << "Expected best_fitness < " << std::scientific << threshold
            << " but got " << res.best_fitness
            << " | x=" << vec_head(res.best_x);
        return make_fail(name, oss.str());
    }

    std::ostringstream ok;
    ok << "best=" << std::scientific << res.best_fitness
       << " x=" << vec_head(res.best_x);
    return make_pass(name, ok.str());
}

static TestCaseResult run_rank_selection() {
    using GA = rcga::RealCodedGA;

    const std::string name = "Rank selection (Sphere MIN)";
    const size_t dim = 10;
    std::vector<rcga::Bounds> b(dim, rcga::Bounds{-5.0, 5.0});

    GA ga;
    auto cfg = base_cfg(dim, b, 20240102);
    cfg.objective = GA::Objective::Minimize;

    cfg.selection = GA::Selection::Rank;
    cfg.population_size = 220;
    cfg.generations = 300;

    ga.set_config(cfg);
    ga.set_fitness([](const std::vector<double>& x) { return sphere(x); });

    auto res = ga.run();

    if (res.best_x.size() != dim) return make_fail(name, "best_x dim mismatch.");
    if (!is_finite(res.best_fitness)) return make_fail(name, "best_fitness not finite.");
    if (!within_bounds(res.best_x, cfg.bounds)) return make_fail(name, "bounds violation.");

    const double threshold = 2e-4;
    if (!(res.best_fitness < threshold)) {
        std::ostringstream oss;
        oss << "Expected best_fitness < " << std::scientific << threshold
            << " but got " << res.best_fitness
            << " | x=" << vec_head(res.best_x);
        return make_fail(name, oss.str());
    }

    std::ostringstream ok;
    ok << "best=" << std::scientific << res.best_fitness
       << " x=" << vec_head(res.best_x);
    return make_pass(name, ok.str());
}

// ✅ AJUSTE 2: BLXAlpha con alpha más conservador + más generaciones
static TestCaseResult run_blxalpha_crossover() {
    using GA = rcga::RealCodedGA;

    const std::string name = "BLX-Alpha crossover (Sphere MIN, alpha=0.2)";
    const size_t dim = 10;
    std::vector<rcga::Bounds> b(dim, rcga::Bounds{-5.0, 5.0});

    GA ga;
    auto cfg = base_cfg(dim, b, 303030);
    cfg.objective = GA::Objective::Minimize;

    cfg.crossover = GA::Crossover::BLXAlpha;
    cfg.blx_alpha = 0.2;

    cfg.population_size = 260;
    cfg.generations = 600;

    ga.set_config(cfg);
    ga.set_fitness([](const std::vector<double>& x) { return sphere(x); });

    auto res = ga.run();

    if (res.best_x.size() != dim) return make_fail(name, "best_x dim mismatch.");
    if (!is_finite(res.best_fitness)) return make_fail(name, "best_fitness not finite.");
    if (!within_bounds(res.best_x, cfg.bounds)) return make_fail(name, "bounds violation.");

    const double threshold = 7e-4;
    if (!(res.best_fitness < threshold)) {
        std::ostringstream oss;
        oss << "Expected best_fitness < " << std::scientific << threshold
            << " but got " << res.best_fitness
            << " | x=" << vec_head(res.best_x);
        return make_fail(name, oss.str());
    }

    std::ostringstream ok;
    ok << "best=" << std::scientific << res.best_fitness
       << " x=" << vec_head(res.best_x);
    return make_pass(name, ok.str());
}

// ✅ AJUSTE 3: UNDX suele explotar si beta/alpha son grandes + mutación alta.
// Reducimos alpha/beta y la prob. de mutación, y damos más generaciones/población.
static TestCaseResult run_undx_crossover() {
    using GA = rcga::RealCodedGA;

    const std::string name = "UNDX crossover (Sphere MIN, tuned)";
    const size_t dim = 10;
    std::vector<rcga::Bounds> b(dim, rcga::Bounds{-5.0, 5.0});

    GA ga;
    auto cfg = base_cfg(dim, b, 404040);
    cfg.objective = GA::Objective::Minimize;

    cfg.crossover = GA::Crossover::UNDX;
    cfg.undx_alpha = 0.30;
    cfg.undx_beta  = 0.15;

    cfg.population_size = 320;
    cfg.generations = 800;

    // Mutación más baja para no “romper” el refinamiento
    cfg.mutation = GA::Mutation::Polynomial;
    cfg.mutation_prob = 0.02; // 2% por gen (en vez de 1/dim=10%)
    cfg.poly_eta = 30.0;

    ga.set_config(cfg);
    ga.set_fitness([](const std::vector<double>& x) { return sphere(x); });

    auto res = ga.run();

    if (res.best_x.size() != dim) return make_fail(name, "best_x dim mismatch.");
    if (!is_finite(res.best_fitness)) return make_fail(name, "best_fitness not finite.");
    if (!within_bounds(res.best_x, cfg.bounds)) return make_fail(name, "bounds violation.");

    const double threshold = 2e-3;
    if (!(res.best_fitness < threshold)) {
        std::ostringstream oss;
        oss << "Expected best_fitness < " << std::scientific << threshold
            << " but got " << res.best_fitness
            << " | x=" << vec_head(res.best_x);
        return make_fail(name, oss.str());
    }

    std::ostringstream ok;
    ok << "best=" << std::scientific << res.best_fitness
       << " x=" << vec_head(res.best_x);
    return make_pass(name, ok.str());
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--verbose") g_verbose = true;
    }

    std::vector<TestCaseResult> results;
    results.reserve(16);

    std::cout << "Running RCGA test suite...\n";
    if (g_verbose) std::cout << "(verbose mode ON)\n";

    results.push_back(run_sphere_min_tournament());
    results.push_back(run_sphere_max_neg_sphere_target_stop());
    results.push_back(run_dim1_edge_case());
    results.push_back(run_per_gene_bounds());
    results.push_back(run_nan_zone_handling());
    results.push_back(run_roulette_selection());
    results.push_back(run_rank_selection());
    results.push_back(run_blxalpha_crossover());
    results.push_back(run_undx_crossover());

    size_t pass_cnt = 0, fail_cnt = 0;

    std::cout << "\n=== RESULTS ===\n";
    for (const auto& r : results) {
        if (r.pass) {
            ++pass_cnt;
            std::cout << "[PASS] " << r.name;
            if (!r.details.empty()) std::cout << " | " << r.details;
            std::cout << "\n";
        } else {
            ++fail_cnt;
            std::cout << "[FAIL] " << r.name;
            if (!r.details.empty()) std::cout << " | " << r.details;
            std::cout << "\n";
        }
    }

    std::cout << "\nSummary: " << pass_cnt << " passed, " << fail_cnt << " failed.\n";
    return (fail_cnt == 0) ? 0 : 1;
}
