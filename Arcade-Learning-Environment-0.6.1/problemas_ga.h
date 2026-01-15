#pragma once

#include <vector>
#include <string>
#include <functional>
#include <limits>
#include "new_ga.h"

using namespace std;

namespace ga {

// forward
class GeneticAlgorithm;


// ------------------------------------------------------------
// 1) Sphere (minimiza)
// ------------------------------------------------------------
class SphereProblem : public IProblem {
public:
    struct Config {
        size_t dimension = 10;
        double min_value = -5.12;
        double max_value =  5.12;
    };

    SphereProblem();
    explicit SphereProblem(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    void attachGA(const GeneticAlgorithm* ga);
    const GeneticAlgorithm* getAttachedGA() const;

    GenomeType genomeType() const override;
    size_t genomeSize() const override;
    Objective objective() const override;

    void getBounds(vector<Bounds>& out_bounds) const override;
    void randomGenome(Genome& out_genome) const override;
    double evaluate(const Genome& genome) const override;
    string describe(const Genome& genome) const override;

private:
    Config cfg_;
    const GeneticAlgorithm* ga_ = nullptr;
};

// ------------------------------------------------------------
// 2) Rastrigin (minimiza)
// ------------------------------------------------------------
class RastriginProblem : public IProblem {
public:
    struct Config {
        size_t dimension = 10;
        double min_value = -5.12;
        double max_value =  5.12;
    };

    RastriginProblem();
    explicit RastriginProblem(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    void attachGA(const GeneticAlgorithm* ga);
    const GeneticAlgorithm* getAttachedGA() const;

    GenomeType genomeType() const override;
    size_t genomeSize() const override;
    Objective objective() const override;

    void getBounds(vector<Bounds>& out_bounds) const override;
    void randomGenome(Genome& out_genome) const override;
    double evaluate(const Genome& genome) const override;
    string describe(const Genome& genome) const override;

private:
    Config cfg_;
    const GeneticAlgorithm* ga_ = nullptr;
};

// ------------------------------------------------------------
// 3) Ackley (minimiza)
// ------------------------------------------------------------
class AckleyProblem : public IProblem {
public:
    struct Config {
        size_t dimension = 10;
        double min_value = -32.768;
        double max_value =  32.768;
    };

    AckleyProblem();
    explicit AckleyProblem(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    void attachGA(const GeneticAlgorithm* ga);
    const GeneticAlgorithm* getAttachedGA() const;

    GenomeType genomeType() const override;
    size_t genomeSize() const override;
    Objective objective() const override;

    void getBounds(vector<Bounds>& out_bounds) const override;
    void randomGenome(Genome& out_genome) const override;
    double evaluate(const Genome& genome) const override;
    string describe(const Genome& genome) const override;

private:
    Config cfg_;
    const GeneticAlgorithm* ga_ = nullptr;
};

// ------------------------------------------------------------
// 4) XOR (binario, maximiza aciertos)
// Genoma: 4 bits -> salida para 00,01,10,11
// ------------------------------------------------------------
class XorProblem : public IProblem {
public:
    struct Config {
        vector<int> expected = {0, 1, 1, 0};
    };

    XorProblem();
    explicit XorProblem(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    void attachGA(const GeneticAlgorithm* ga);
    const GeneticAlgorithm* getAttachedGA() const;

    GenomeType genomeType() const override;
    size_t genomeSize() const override;
    Objective objective() const override;

    void getBounds(vector<Bounds>& out_bounds) const override;
    void randomGenome(Genome& out_genome) const override;
    double evaluate(const Genome& genome) const override;
    string describe(const Genome& genome) const override;

private:
    Config cfg_;
    const GeneticAlgorithm* ga_ = nullptr;
};

// ------------------------------------------------------------
// 5) OneMax (binario, maximiza nº de unos)
// ------------------------------------------------------------
class OneMaxProblem : public IProblem {
public:
    struct Config {
        size_t n_bits = 100;
    };

    OneMaxProblem();
    explicit OneMaxProblem(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    void attachGA(const GeneticAlgorithm* ga);
    const GeneticAlgorithm* getAttachedGA() const;

    GenomeType genomeType() const override;
    size_t genomeSize() const override;
    Objective objective() const override;

    void getBounds(vector<Bounds>& out_bounds) const override;
    void randomGenome(Genome& out_genome) const override;
    double evaluate(const Genome& genome) const override;
    string describe(const Genome& genome) const override;

private:
    Config cfg_;
    const GeneticAlgorithm* ga_ = nullptr;
};

// ------------------------------------------------------------
// 6) ALE/Atari (caja negra por callbacks)
// ------------------------------------------------------------
class AleAtariProblem : public IProblem {
public:
    using InitFunction = function<void(const AleAtariProblem& self, Genome& out_genome)>;
    using FitnessFunction = function<double(const AleAtariProblem& self, const Genome& genome)>;
    using DescribeFunction = function<string(const AleAtariProblem& self, const Genome& genome)>;

    struct Config {
        GenomeType type = GenomeType::Real;
        size_t genome_size = 0;
        Objective obj = Objective::Maximize;

        vector<Bounds> bounds;

        int episodes_per_eval = 1;
        int max_steps_per_episode = 18000;

        InitFunction init_fn;
        FitnessFunction fitness_fn;
        DescribeFunction describe_fn;
    };

    AleAtariProblem();
    explicit AleAtariProblem(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    void attachGA(const GeneticAlgorithm* ga);
    const GeneticAlgorithm* getAttachedGA() const;

    int getEpisodesPerEval() const;
    int getMaxStepsPerEpisode() const;

    GenomeType genomeType() const override;
    size_t genomeSize() const override;
    Objective objective() const override;

    void getBounds(vector<Bounds>& out_bounds) const override;
    void randomGenome(Genome& out_genome) const override;
    double evaluate(const Genome& genome) const override;
    string describe(const Genome& genome) const override;

private:
    Config cfg_;
    const GeneticAlgorithm* ga_ = nullptr;

private:
    void fillDefaultBoundsIfNeeded(vector<Bounds>& b) const;
    double worstFitness() const;
};
class TspRandomKeysProblem : public IProblem {
public:
    // Ciudad simple (coordenadas 2D)
    struct Point2D {
        double x = 0.0;
        double y = 0.0;
    };

    struct Config {
        // Si use_distance_matrix = false, se usan estas coordenadas
        vector<Point2D> cities;

        // Si use_distance_matrix = true, se usa esta matriz NxN
        vector<vector<double>> dist_matrix;

        bool use_distance_matrix = false; // false -> coords, true -> matrix
        bool closed_tour = true;          // true -> vuelve a la primera ciudad
    };

public:
    // Constructores
    TspRandomKeysProblem();
    explicit TspRandomKeysProblem(const Config& cfg);

    // Configuración
    void setConfig(const Config& cfg);
    Config getConfig() const;

    // Conexión opcional al GA (para random reproducible)
    void attachGA(const GeneticAlgorithm* ga);
    const GeneticAlgorithm* getAttachedGA() const;

    // IProblem
    GenomeType genomeType() const override;
    size_t genomeSize() const override;
    Objective objective() const override;

    void getBounds(vector<Bounds>& out_bounds) const override;
    void randomGenome(Genome& out_genome) const override;
    double evaluate(const Genome& genome) const override;
    string describe(const Genome& genome) const override;

private:
    // Devuelve distancia entre dos ciudades (según config)
    double distanceBetween(int a, int b) const;

    // Convierte random keys -> orden de ciudades (tour)
    void decodeTour(const Genome& genome, vector<int>& out_tour) const;

    // Calcula distancia total de un tour
    double tourLength(const vector<int>& tour) const;

private:
    Config cfg_;
    const GeneticAlgorithm* ga_ = nullptr;
};
} // namespace ga
