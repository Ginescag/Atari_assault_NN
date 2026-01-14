#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <chrono>
#include <random>
#include <cstdint>

namespace rcga {

// Cotas tipo “box constraints” por gen
struct Bounds {
    double lo = -1.0;
    double hi =  1.0;
};

inline bool isfinite(double v) { return std::isfinite(v) != 0; }
inline bool isfinite(long double v) { return std::isfinite(v) != 0; }

// clamp robusto ante NaN/Inf -> devuelve el punto medio del intervalo
inline double clamp(double v, double lo, double hi) {
    if (!std::isfinite(v)) {
        // evitar overflow de (lo + hi) cuando son muy grandes
        return 0.5 * lo + 0.5 * hi;
    }
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

class RealCodedGA {
public:
    enum class Objective { Minimize, Maximize };
    enum class Selection { Tournament, Roulette, Rank };
    enum class Crossover { SBX, BLXAlpha, UNDX, Uniform, None };
    enum class Mutation  { Polynomial, Gaussian, None };

    using Vec       = std::vector<double>;
    using FitnessFn = std::function<double(const Vec&)>;

    struct Config {
        // Problema
        std::size_t dim = 0;
        std::vector<Bounds> bounds;   // si está vacío -> usar uniform_bounds
        Bounds uniform_bounds{-1.0, 1.0};

        // GA
        std::size_t population_size = 200;
        std::size_t generations     = 200;
        Objective objective = Objective::Minimize;

        // Selección
        Selection selection = Selection::Tournament;
        std::size_t tournament_k = 3;

        // Crossover
        Crossover crossover = Crossover::SBX;
        double crossover_prob = 0.9;

        // Mutación
        Mutation mutation = Mutation::Polynomial;
        double mutation_prob = -1.0; // <0 => 1/dim por gen

        // Parámetros de operadores
        double sbx_eta     = 20.0; // eta_c (>=0)
        double blx_alpha   = 0.5;  // alpha (>=0)
        double undx_alpha  = 0.5;  // (>=0)
        double undx_beta   = 0.35; // (>=0)
        double poly_eta    = 20.0; // eta_m (>=0)
        double gauss_sigma = 0.05; // sigma relativo >=0 (multiplica hi-lo)

        // Reemplazo / elitismo
        std::size_t elite_count = 1;

        // Stopping
        bool use_target_fitness = false;
        double target_fitness   = 0.0;  // raw fitness
        std::size_t stall_generations = 0; // 0 => desactivado
        double stall_epsilon = 1e-12;       // >=0

        // RNG
        std::uint64_t seed = 0; // 0 => random_device
    };

    struct Result {
        Vec best_x;
        double best_fitness = std::numeric_limits<double>::quiet_NaN(); // raw
        std::size_t best_generation = 0;
        std::vector<double> history_best; // raw por generación
        std::vector<double> history_mean; // raw por generación
    };

    RealCodedGA() : RealCodedGA(Config{}) {}
    explicit RealCodedGA(Config cfg) : cfg_(std::move(cfg)) { reseed(cfg_.seed); }

    void set_config(Config cfg) {
        cfg_ = std::move(cfg);
        reseed(cfg_.seed);
    }
    const Config& config() const { return cfg_; }

    void set_fitness(FitnessFn fn) { fitness_fn_ = std::move(fn); }

    // (Opcional) población inicial (size == population_size)
    void set_initial_population(std::vector<Vec> init) { initial_population_ = std::move(init); }

    // (Opcional) callback por generación: (gen, best_raw, mean_raw)
    void set_progress_callback(std::function<void(std::size_t, double, double)> cb) {
        progress_cb_ = std::move(cb);
    }

    void reseed(std::uint64_t seed) {
        if (seed == 0) {
            std::random_device rd;
            std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
            rng_.seed(seq);
        } else {
            rng_.seed(seed);
        }
    }

    Result run() {
        validate_config_();
        if (!fitness_fn_) throw std::runtime_error("RealCodedGA: fitness function not set.");

        // ---- Init population ----
        std::vector<Individual> pop;
        pop.reserve(cfg_.population_size);

        if (!initial_population_.empty()) {
            if (initial_population_.size() != cfg_.population_size)
                throw std::runtime_error("RealCodedGA: initial_population size != population_size.");

            for (const auto& x_in : initial_population_) {
                if (x_in.size() != cfg_.dim)
                    throw std::runtime_error("RealCodedGA: wrong dim in initial individual.");

                Vec x = x_in;
                clamp_vec_to_bounds_(x);

                pop.push_back(Individual{std::move(x), 0.0, std::numeric_limits<double>::quiet_NaN()});
            }
        } else {
            for (std::size_t i = 0; i < cfg_.population_size; ++i)
                pop.push_back(Individual{random_vec_(), 0.0, std::numeric_limits<double>::quiet_NaN()});
        }

        evaluate_population_(pop, 0);

        Result res;
        res.history_best.reserve(cfg_.generations + 1);
        res.history_mean.reserve(cfg_.generations + 1);

        auto [b0, raw0] = best_of_(pop);
        res.best_x = pop[b0].x;
        res.best_fitness = raw0;
        res.best_generation = 0;

        double best_raw_so_far = raw0;
        std::size_t stall = 0;
        bool stopped_early = false;

        for (std::size_t gen = 0; gen < cfg_.generations; ++gen) {
            const auto stats = stats_of_(pop);
            res.history_best.push_back(stats.best_raw);
            res.history_mean.push_back(stats.mean_raw);
            if (progress_cb_) progress_cb_(gen, stats.best_raw, stats.mean_raw);

            if (cfg_.use_target_fitness && is_target_reached_(stats.best_raw)) {
                stopped_early = true;
                break;
            }

            if (cfg_.stall_generations > 0) {
                if (is_improvement_(stats.best_raw, best_raw_so_far)) {
                    best_raw_so_far = stats.best_raw;
                    stall = 0;
                } else {
                    if (++stall >= cfg_.stall_generations) {
                        stopped_early = true;
                        break;
                    }
                }
            }

            // ---- Precompute selection cache for THIS generation ----
            build_selection_cache_(pop);

            // ---- Next generation ----
            std::vector<Individual> next;
            next.reserve(cfg_.population_size);

            // Elites SOLO finitos (copiados con su fitness; NO se reevaluarán)
            std::size_t elites_to_copy = 0;
            if (cfg_.elite_count > 0) {
                const auto elite_idx = top_k_indices_finite_(pop, cfg_.elite_count);
                elites_to_copy = elite_idx.size();
                for (std::size_t idx : elite_idx) next.push_back(pop[idx]);
            }

            while (next.size() < cfg_.population_size) {
                const Individual& p1 = select_parent_cached_(pop);
                const Individual& p2 = select_parent_cached_(pop);

                Vec c1 = p1.x;
                Vec c2 = p2.x;

                if (rand01_() < cfg_.crossover_prob && cfg_.crossover != Crossover::None) {
                    apply_crossover_(p1.x, p2.x, c1, c2, pop);
                }

                apply_mutation_(c1);
                apply_mutation_(c2);

                // seguridad extra: mantener siempre dentro de bounds
                clamp_vec_to_bounds_(c1);
                clamp_vec_to_bounds_(c2);

                next.push_back(Individual{std::move(c1), 0.0, std::numeric_limits<double>::quiet_NaN()});
                if (next.size() < cfg_.population_size)
                    next.push_back(Individual{std::move(c2), 0.0, std::numeric_limits<double>::quiet_NaN()});
            }

            // Reevaluamos SOLO los no-elite
            evaluate_population_(next, elites_to_copy);
            pop.swap(next);

            // Update global best
            auto [bi, br] = best_of_(pop);
            if (is_better_raw_(br, res.best_fitness)) {
                res.best_x = pop[bi].x;
                res.best_fitness = br;
                res.best_generation = gen + 1;
            }
        }

        if (!stopped_early) {
            if (res.history_best.size() < cfg_.generations + 1) {
                const auto stats = stats_of_(pop);
                res.history_best.push_back(stats.best_raw);
                res.history_mean.push_back(stats.mean_raw);
            }
        }

        return res;
    }

private:
    struct Individual {
        Vec x;
        double fit_internal = 0.0; // siempre minimizamos internamente
        double fit_raw = std::numeric_limits<double>::quiet_NaN(); // raw original
    };

    struct Stats {
        double best_raw;
        double mean_raw;
    };

    // Cache de selección por generación (evita ordenar / recomputar pesos en cada parent pick)
    struct SelectionCache {
        const std::vector<Individual>* pop_ptr = nullptr;
        Selection kind = Selection::Tournament;

        // Para Roulette:
        std::vector<double> cumw;
        double totalw = 0.0;

        // Para Rank (SOLO finitos):
        std::vector<std::size_t> idx_sorted; // índices finitos ordenados por fit_internal asc
        std::vector<double> cumrank;
        double totalrank = 0.0;

        void clear() {
            pop_ptr = nullptr;
            cumw.clear(); totalw = 0.0;
            idx_sorted.clear(); cumrank.clear(); totalrank = 0.0;
        }
    };

    Config cfg_;
    FitnessFn fitness_fn_;
    std::vector<Vec> initial_population_;
    std::function<void(std::size_t, double, double)> progress_cb_;
    std::mt19937_64 rng_{0xC0FFEEULL};

    SelectionCache sel_cache_;

    void validate_config_() const {
        if (cfg_.dim == 0) throw std::runtime_error("RealCodedGA: cfg.dim must be > 0.");
        if (cfg_.population_size < 2) throw std::runtime_error("RealCodedGA: population_size must be >= 2.");
        if (!cfg_.bounds.empty() && cfg_.bounds.size() != cfg_.dim)
            throw std::runtime_error("RealCodedGA: bounds.size() must be 0 or dim.");

        // si elite_count == population_size, el GA no genera hijos
        if (cfg_.elite_count >= cfg_.population_size)
            throw std::runtime_error("RealCodedGA: elite_count must be < population_size (otherwise no offspring are produced).");

        auto require_finite = [&](double v, const char* name) {
            if (!isfinite(v)) throw std::runtime_error(std::string("RealCodedGA: ") + name + " must be finite.");
        };
        auto require_prob01 = [&](double v, const char* name) {
            require_finite(v, name);
            if (v < 0.0 || v > 1.0)
                throw std::runtime_error(std::string("RealCodedGA: ") + name + " must be in [0,1].");
        };
        auto require_nonneg = [&](double v, const char* name) {
            require_finite(v, name);
            if (v < 0.0)
                throw std::runtime_error(std::string("RealCodedGA: ") + name + " must be >= 0.");
        };

        require_prob01(cfg_.crossover_prob, "crossover_prob");

        require_finite(cfg_.mutation_prob, "mutation_prob");
        if (cfg_.mutation_prob >= 0.0 && cfg_.mutation_prob > 1.0)
            throw std::runtime_error("RealCodedGA: mutation_prob must be in [0,1] or <0 for default.");

        require_nonneg(cfg_.sbx_eta,     "sbx_eta");
        require_nonneg(cfg_.blx_alpha,   "blx_alpha");
        require_nonneg(cfg_.undx_alpha,  "undx_alpha");
        require_nonneg(cfg_.undx_beta,   "undx_beta");
        require_nonneg(cfg_.poly_eta,    "poly_eta");
        require_nonneg(cfg_.gauss_sigma, "gauss_sigma");

        require_nonneg(cfg_.stall_epsilon, "stall_epsilon");

        if (cfg_.use_target_fitness) {
            require_finite(cfg_.target_fitness, "target_fitness");
        }

        if (cfg_.selection == Selection::Tournament) {
            if (cfg_.tournament_k < 2)
                throw std::runtime_error("RealCodedGA: tournament_k must be >= 2.");
            if (cfg_.tournament_k > cfg_.population_size)
                throw std::runtime_error("RealCodedGA: tournament_k must be <= population_size.");
        }

        auto check_bounds = [&](const Bounds& b) {
            if (!isfinite(b.lo) || !isfinite(b.hi))
                throw std::runtime_error("RealCodedGA: bounds must be finite.");
            if (b.lo > b.hi)
                throw std::runtime_error("RealCodedGA: bounds invalid (lo > hi).");
        };

        if (cfg_.bounds.empty()) {
            check_bounds(cfg_.uniform_bounds);
        } else {
            for (const auto& b : cfg_.bounds) check_bounds(b);
        }
    }

    Bounds bounds_(std::size_t i) const { return cfg_.bounds.empty() ? cfg_.uniform_bounds : cfg_.bounds[i]; }

    void clamp_vec_to_bounds_(Vec& x) const {
        for (std::size_t i = 0; i < cfg_.dim; ++i) {
            const auto b = bounds_(i);
            x[i] = clamp(x[i], b.lo, b.hi);
        }
    }

    double rand01_() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng_); }
    double randu_(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng_); }

    // normal_distribution con stddev<=0 es UB -> devolvemos mean (ruido 0)
    double randn_(double mean, double stddev) {
        if (!(stddev > 0.0) || !std::isfinite(stddev)) return mean;
        return std::normal_distribution<double>(mean, stddev)(rng_);
    }

    std::size_t randi_(std::size_t lo, std::size_t hi_incl) {
        return std::uniform_int_distribution<std::size_t>(lo, hi_incl)(rng_);
    }

    std::size_t random_finite_index_(const std::vector<Individual>& pop) {
        // asume que existe al menos uno finito (lo garantizamos en evaluate_population_)
        for (int t = 0; t < 64; ++t) {
            const std::size_t i = randi_(0, pop.size() - 1);
            if (std::isfinite(pop[i].fit_internal)) return i;
        }
        // fallback (muy raro): búsqueda lineal
        for (std::size_t i = 0; i < pop.size(); ++i) {
            if (std::isfinite(pop[i].fit_internal)) return i;
        }
        return 0;
    }

    Vec random_vec_() {
        Vec x(cfg_.dim);
        for (std::size_t i = 0; i < cfg_.dim; ++i) {
            const auto b = bounds_(i);
            x[i] = randu_(b.lo, b.hi);
        }
        return x;
    }

    double raw_to_internal_(double raw) const {
        return (cfg_.objective == Objective::Minimize) ? raw : -raw;
    }

    bool is_better_raw_(double a_raw, double b_raw) const {
        if (!isfinite(b_raw)) return true;
        if (!isfinite(a_raw)) return false;
        return (cfg_.objective == Objective::Minimize) ? (a_raw < b_raw) : (a_raw > b_raw);
    }

    bool is_improvement_(double now, double prev) const {
        if (!isfinite(prev)) return true;
        if (!isfinite(now))  return false;
        return (cfg_.objective == Objective::Minimize)
            ? ((prev - now) > cfg_.stall_epsilon)
            : ((now - prev) > cfg_.stall_epsilon);
    }

    bool is_target_reached_(double best_raw) const {
        if (!isfinite(best_raw)) return false;
        return (cfg_.objective == Objective::Minimize)
            ? (best_raw <= cfg_.target_fitness)
            : (best_raw >= cfg_.target_fitness);
    }

    void evaluate_population_(std::vector<Individual>& pop, std::size_t start_index) {
        if (start_index > pop.size()) start_index = pop.size();

        for (std::size_t i = start_index; i < pop.size(); ++i) {
            auto& ind = pop[i];

            // seguridad: mantener en bounds antes de evaluar
            clamp_vec_to_bounds_(ind.x);

            const double raw = fitness_fn_(ind.x);
            ind.fit_raw = raw;

            if (!isfinite(raw)) {
                ind.fit_internal = std::numeric_limits<double>::infinity();
            } else {
                ind.fit_internal = raw_to_internal_(raw);
            }
        }

        std::size_t finite_cnt = 0;
        for (const auto& ind : pop) {
            if (isfinite(ind.fit_raw)) ++finite_cnt;
        }
        if (finite_cnt == 0) {
            throw std::runtime_error("RealCodedGA: all individuals returned non-finite fitness (NaN/Inf).");
        }
    }

    std::pair<std::size_t, double> best_of_(const std::vector<Individual>& pop) const {
        std::size_t best = 0;
        for (std::size_t i = 1; i < pop.size(); ++i) {
            if (pop[i].fit_internal < pop[best].fit_internal) best = i;
        }
        return {best, pop[best].fit_raw};
    }

    Stats stats_of_(const std::vector<Individual>& pop) const {
        auto [bi, br] = best_of_(pop);
        (void)bi;

        double sum = 0.0;
        std::size_t cnt = 0;
        for (const auto& ind : pop) {
            if (isfinite(ind.fit_raw)) {
                sum += ind.fit_raw;
                ++cnt;
            }
        }
        const double mean_raw = (cnt > 0) ? (sum / static_cast<double>(cnt))
                                          : std::numeric_limits<double>::quiet_NaN();

        return Stats{br, mean_raw};
    }

    // elites SOLO finitos
    std::vector<std::size_t> top_k_indices_finite_(const std::vector<Individual>& pop, std::size_t k) const {
        std::vector<std::size_t> idx;
        idx.reserve(pop.size());
        for (std::size_t i = 0; i < pop.size(); ++i) {
            if (std::isfinite(pop[i].fit_internal)) idx.push_back(i);
        }
        if (idx.empty()) return {};
        if (k > idx.size()) k = idx.size();
        std::partial_sort(
            idx.begin(),
            idx.begin() + static_cast<std::ptrdiff_t>(k),
            idx.end(),
            [&](std::size_t a, std::size_t b) { return pop[a].fit_internal < pop[b].fit_internal; }
        );
        idx.resize(k);
        return idx;
    }

    // ---- Selection (con caché) ----
    void build_selection_cache_(const std::vector<Individual>& pop) {
        sel_cache_.clear();
        sel_cache_.pop_ptr = &pop;
        sel_cache_.kind = cfg_.selection;

        if (cfg_.selection == Selection::Roulette) {
            // Minimización interna: menor fit_internal => mejor.
            double best = std::numeric_limits<double>::infinity();
            double worst = -std::numeric_limits<double>::infinity();
            bool any = false;

            for (const auto& ind : pop) {
                if (!std::isfinite(ind.fit_internal)) continue;
                any = true;
                best = std::min(best, ind.fit_internal);
                worst = std::max(worst, ind.fit_internal);
            }
            if (!any) return; // fallback

            const long double span_ld =
                static_cast<long double>(worst) - static_cast<long double>(best);
            const double eps = 1e-12;

            sel_cache_.cumw.resize(pop.size(), 0.0);
            double acc = 0.0;

            const bool span_ok = isfinite(span_ld) && (span_ld > 0.0L);

            for (std::size_t i = 0; i < pop.size(); ++i) {
                const double fi = pop[i].fit_internal;
                double w = 0.0;

                if (!std::isfinite(fi)) {
                    w = 0.0; // ✅ inválidos no seleccionables
                } else if (!span_ok) {
                    w = 1.0; // casi uniforme entre finitos
                } else {
                    const long double wld =
                        (static_cast<long double>(worst) - static_cast<long double>(fi)) / span_ld;

                    w = (isfinite(wld) && wld >= 0.0L) ? static_cast<double>(wld) : 0.0;
                    w += eps;
                }

                acc += w;
                sel_cache_.cumw[i] = acc;
            }

            sel_cache_.totalw = acc;
            return;
        }

        if (cfg_.selection == Selection::Rank) {
            // ✅ FIX: rank SOLO sobre finitos
            sel_cache_.idx_sorted.clear();
            sel_cache_.idx_sorted.reserve(pop.size());
            for (std::size_t i = 0; i < pop.size(); ++i) {
                if (std::isfinite(pop[i].fit_internal)) sel_cache_.idx_sorted.push_back(i);
            }
            const std::size_t N = sel_cache_.idx_sorted.size();
            if (N == 0) return; // fallback

            std::sort(sel_cache_.idx_sorted.begin(), sel_cache_.idx_sorted.end(),
                [&](std::size_t a, std::size_t b) { return pop[a].fit_internal < pop[b].fit_internal; });

            sel_cache_.cumrank.resize(N, 0.0);
            double acc = 0.0;
            for (std::size_t r = 0; r < N; ++r) {
                acc += static_cast<double>(N - r); // mejor recibe N, peor 1
                sel_cache_.cumrank[r] = acc;
            }
            sel_cache_.totalrank = acc;
            return;
        }

        // Tournament: no cache needed
    }

    const Individual& select_parent_cached_(const std::vector<Individual>& pop) {
        switch (cfg_.selection) {
            case Selection::Tournament:
                return tournament_select_(pop);

            case Selection::Roulette: {
                if (sel_cache_.pop_ptr != &pop || sel_cache_.kind != Selection::Roulette ||
                    sel_cache_.cumw.empty() || !(sel_cache_.totalw > 0.0) || !std::isfinite(sel_cache_.totalw)) {
                    return pop[random_finite_index_(pop)];
                }

                double u = rand01_();
                if (u >= 1.0) u = std::nextafter(1.0, 0.0);
                const double r = u * sel_cache_.totalw;

                auto it = std::upper_bound(sel_cache_.cumw.begin(), sel_cache_.cumw.end(), r);
                std::size_t idx = (it == sel_cache_.cumw.end())
                    ? (pop.size() - 1)
                    : static_cast<std::size_t>(it - sel_cache_.cumw.begin());

                if (!std::isfinite(pop[idx].fit_internal)) return pop[random_finite_index_(pop)];
                return pop[idx];
            }

            case Selection::Rank: {
                if (sel_cache_.pop_ptr != &pop || sel_cache_.kind != Selection::Rank ||
                    sel_cache_.idx_sorted.empty() || sel_cache_.cumrank.empty() ||
                    !(sel_cache_.totalrank > 0.0) || !std::isfinite(sel_cache_.totalrank)) {
                    return pop[random_finite_index_(pop)];
                }

                double u = rand01_();
                if (u >= 1.0) u = std::nextafter(1.0, 0.0);
                const double r = u * sel_cache_.totalrank;

                auto it = std::lower_bound(sel_cache_.cumrank.begin(), sel_cache_.cumrank.end(), r);
                std::size_t ridx = (it == sel_cache_.cumrank.end())
                    ? (sel_cache_.idx_sorted.size() - 1)
                    : static_cast<std::size_t>(it - sel_cache_.cumrank.begin());

                return pop[ sel_cache_.idx_sorted[ridx] ];
            }
        }
        return tournament_select_(pop);
    }

    const Individual& tournament_select_(const std::vector<Individual>& pop) {
        // ✅ FIX: arrancar desde un candidato finito
        std::size_t best = random_finite_index_(pop);
        for (std::size_t i = 1; i < cfg_.tournament_k; ++i) {
            const std::size_t j = randi_(0, pop.size() - 1);
            if (pop[j].fit_internal < pop[best].fit_internal) best = j;
        }
        return pop[best];
    }

    // ---- Crossover ----
    void apply_crossover_(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2,
                          const std::vector<Individual>& pop) {
        switch (cfg_.crossover) {
            case Crossover::SBX:      sbx_(p1, p2, c1, c2); break;
            case Crossover::BLXAlpha: blx_alpha_(p1, p2, c1, c2); break;
            case Crossover::UNDX:     undx_(p1, p2, c1, c2, pop); break;
            case Crossover::Uniform:  uniform_xover_(p1, p2, c1, c2); break;
            case Crossover::None:     break;
        }
    }

    void uniform_xover_(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2) {
        for (std::size_t i = 0; i < cfg_.dim; ++i) {
            if (rand01_() < 0.5) { c1[i] = p1[i]; c2[i] = p2[i]; }
            else                 { c1[i] = p2[i]; c2[i] = p1[i]; }
        }
    }

    void sbx_(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2) {
        const double eta = cfg_.sbx_eta; // ya validado >=0

        for (std::size_t i = 0; i < cfg_.dim; ++i) {
            const auto b = bounds_(i);
            const double x1 = clamp(p1[i], b.lo, b.hi);
            const double x2 = clamp(p2[i], b.lo, b.hi);

            if (rand01_() > 0.5 || std::fabs(x1 - x2) < 1e-14) {
                c1[i] = x1; c2[i] = x2; continue;
            }

            double u = rand01_();
            if (u <= 0.0) u = std::numeric_limits<double>::min();
            if (u >= 1.0) u = std::nextafter(1.0, 0.0);

            double beta_q;
            if (u <= 0.5) beta_q = std::pow(2.0 * u, 1.0 / (eta + 1.0));
            else          beta_q = std::pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta + 1.0));

            const double child1 = 0.5 * ((1.0 + beta_q) * x1 + (1.0 - beta_q) * x2);
            const double child2 = 0.5 * ((1.0 - beta_q) * x1 + (1.0 + beta_q) * x2);

            c1[i] = clamp(child1, b.lo, b.hi);
            c2[i] = clamp(child2, b.lo, b.hi);
        }
    }

    void blx_alpha_(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2) {
        const double a = cfg_.blx_alpha; // ya validado >=0
        for (std::size_t i = 0; i < cfg_.dim; ++i) {
            const auto b = bounds_(i);
            const double lo = std::min(p1[i], p2[i]);
            const double hi = std::max(p1[i], p2[i]);
            const double d  = hi - lo;
            const double L  = lo - a * d;
            const double H  = hi + a * d;
            c1[i] = clamp(randu_(L, H), b.lo, b.hi);
            c2[i] = clamp(randu_(L, H), b.lo, b.hi);
        }
    }

    static double dot_(const Vec& a, const Vec& b) {
        double s = 0.0; for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s;
    }
    static double norm_(const Vec& a) { return std::sqrt(dot_(a, a)); }
    static Vec sub_(const Vec& a, const Vec& b) {
        Vec r(a.size()); for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i]; return r;
    }
    static Vec add_(const Vec& a, const Vec& b) {
        Vec r(a.size()); for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i]; return r;
    }
    static Vec mul_(const Vec& a, double s) {
        Vec r(a.size()); for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] * s; return r;
    }
    static void add_inplace_(Vec& a, const Vec& b, double s = 1.0) {
        for (std::size_t i = 0; i < a.size(); ++i) a[i] += s * b[i];
    }

    void undx_(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2, const std::vector<Individual>& pop) {
        if (cfg_.dim < 2 || pop.size() < 3) { sbx_(p1, p2, c1, c2); return; }

        const Vec* p1ptr = &p1;
        const Vec* p2ptr = &p2;

        const Vec* p3ptr = nullptr;
        for (std::size_t attempt = 0; attempt < 64; ++attempt) {
            const std::size_t idx = randi_(0, pop.size() - 1);
            const Vec* cand = &pop[idx].x;
            if (cand != p1ptr && cand != p2ptr) { p3ptr = cand; break; }
        }
        if (!p3ptr) { sbx_(p1, p2, c1, c2); return; }
        const Vec& p3 = *p3ptr;

        const Vec d12 = sub_(p2, p1);
        const double d1 = norm_(d12);
        if (d1 < 1e-14) { c1 = p1; c2 = p2; return; }

        Vec e1 = mul_(d12, 1.0 / d1);

        const Vec d13 = sub_(p3, p1);
        const double proj = dot_(d13, e1);
        Vec closest = add_(p1, mul_(e1, proj));
        const double d2 = norm_(sub_(p3, closest));

        const double sigma1 = cfg_.undx_alpha * d1;
        const double sigma2 = cfg_.undx_beta * d2 / std::sqrt(static_cast<double>(cfg_.dim));

        std::vector<Vec> basis;
        basis.reserve(cfg_.dim);
        basis.push_back(e1);

        const std::size_t max_attempts_per_vec = 64;

        for (std::size_t k = 1; k < cfg_.dim; ++k) {
            bool ok = false;
            for (std::size_t attempt = 0; attempt < max_attempts_per_vec; ++attempt) {
                Vec v(cfg_.dim);
                for (std::size_t i = 0; i < cfg_.dim; ++i) v[i] = randn_(0.0, 1.0);

                for (const auto& bvec : basis) {
                    const double pr = dot_(v, bvec);
                    add_inplace_(v, bvec, -pr);
                }

                const double nv = norm_(v);
                if (nv < 1e-10) continue;

                for (double& t : v) t /= nv;
                basis.push_back(std::move(v));
                ok = true;
                break;
            }

            if (!ok) {
                sbx_(p1, p2, c1, c2);
                return;
            }
        }

        const Vec m = mul_(add_(p1, p2), 0.5);

        Vec off(cfg_.dim, 0.0);
        add_inplace_(off, basis[0], randn_(0.0, sigma1));
        for (std::size_t k = 1; k < cfg_.dim; ++k)
            add_inplace_(off, basis[k], randn_(0.0, sigma2));

        c1 = m; c2 = m;
        add_inplace_(c1, off, +1.0);
        add_inplace_(c2, off, -1.0);

        clamp_vec_to_bounds_(c1);
        clamp_vec_to_bounds_(c2);
    }

    // ---- Mutation ----
    void apply_mutation_(Vec& x) {
        if (cfg_.mutation == Mutation::None) return;
        const double pm = (cfg_.mutation_prob < 0.0)
            ? (1.0 / static_cast<double>(cfg_.dim))
            : cfg_.mutation_prob;

        const double pmc = (pm < 0.0) ? 0.0 : (pm > 1.0 ? 1.0 : pm);

        switch (cfg_.mutation) {
            case Mutation::Polynomial: polynomial_mutation_(x, pmc); break;
            case Mutation::Gaussian:   gaussian_mutation_(x, pmc); break;
            case Mutation::None:       break;
        }
    }

    void polynomial_mutation_(Vec& x, double pm) {
        const double eta = cfg_.poly_eta;

        for (std::size_t i = 0; i < cfg_.dim; ++i) {
            if (rand01_() > pm) continue;

            const auto b = bounds_(i);
            const double xl = b.lo;
            const double xu = b.hi;
            const double xi = clamp(x[i], xl, xu);

            const double span = xu - xl;
            if (!(span > 1e-18) || !std::isfinite(span)) { x[i] = xl; continue; }

            const double delta1 = (xi - xl) / span;
            const double delta2 = (xu - xi) / span;
            double u = rand01_();
            if (u <= 0.0) u = std::numeric_limits<double>::min();
            if (u >= 1.0) u = std::nextafter(1.0, 0.0);

            double deltaq = 0.0;
            const double mut_pow = 1.0 / (eta + 1.0);

            if (u <= 0.5) {
                const double xy = 1.0 - delta1;
                double val = 2.0 * u + (1.0 - 2.0 * u) * std::pow(xy, eta + 1.0);
                //FIX: evitar val < 0 por redondeos extremos
                if (val < 0.0) val = 0.0;
                deltaq = std::pow(val, mut_pow) - 1.0;
            } else {
                const double xy = 1.0 - delta2;
                double val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * std::pow(xy, eta + 1.0);
                if (val < 0.0) val = 0.0;
                deltaq = 1.0 - std::pow(val, mut_pow);
            }

            x[i] = clamp(xi + deltaq * span, xl, xu);
        }
    }

    void gaussian_mutation_(Vec& x, double pm) {
        const double sigma_rel = cfg_.gauss_sigma;
        for (std::size_t i = 0; i < cfg_.dim; ++i) {
            if (rand01_() > pm) continue;
            const auto b = bounds_(i);
            const double span = (b.hi - b.lo);
            const double sigma = sigma_rel * span;
            x[i] = clamp(x[i] + randn_(0.0, sigma), b.lo, b.hi);
        }
    }
};

} // namespace rcga
