// new_ga.h
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace ga {

using namespace std;

// ---------------------------
// Tipos básicos
// ---------------------------

enum class GenomeType {
    Binary,
    Real
};

enum class Objective {
    Maximize,
    Minimize
};

struct Bounds {
    double lo = -1.0;
    double hi =  1.0;
};

// ---------------------------
// Genome
// ---------------------------

struct Genome {
    GenomeType type = GenomeType::Real;

    // Se usa uno u otro según type
    vector<int> bits;
    vector<double> reals;

    Genome();
    Genome(GenomeType t);

    void clear();

    GenomeType getType() const;
    void setType(GenomeType t);

    size_t size() const;

    void resizeBinary(size_t n_bits);
    void resizeReal(size_t n_reals);

    int getBit(size_t i) const;
    void setBit(size_t i, int v);

    double getReal(size_t i) const;
    void setReal(size_t i, double v);
};

// ---------------------------
// Individual
// ---------------------------

struct Individual {
    Genome genome;

    double fitness = 0.0;
    bool fitness_valid = false;//me dice si el fitness es fiable o si esta desactualizado

    Individual();
    Individual(const Genome& g);

    //Cada vez que el genoma cambia (mutación o cruce), el fitness anterior deja de corresponder
    //esta función hace que el resto del sistema sepa hay que reevaluar
    void invalidateFitness();
};

// ---------------------------
// Interfaces del GA
// ---------------------------

class IProblem {
public:
    virtual ~IProblem() = default;

    virtual GenomeType genomeType() const = 0;
    virtual Objective objective() const = 0;

    virtual void getBounds(vector<Bounds>& out_bounds) const = 0;
    virtual void randomGenome(Genome& out_genome) const = 0;

    virtual double evaluate(const Genome& genome) const = 0;
    virtual size_t genomeSize() const = 0;
    virtual string describe(const Genome& genome) const = 0;


};

class GeneticAlgorithm;

class ISelection {
public:
    virtual ~ISelection() = default;
    virtual size_t selectIndex(const GeneticAlgorithm& ga, const vector<Individual>& population) = 0;
    virtual void reset() = 0;
};

class ICrossover {
public:
    virtual ~ICrossover() = default;
    virtual void crossover(const GeneticAlgorithm& ga,
                           const Genome& parent_a,
                           const Genome& parent_b,
                           Genome& out_child_a,
                           Genome& out_child_b) = 0;
    virtual void reset() = 0;
};

class IMutation {
public:
    virtual ~IMutation() = default;
    virtual void mutate(const GeneticAlgorithm& ga, Genome& inout_genome, const vector<Bounds>& bounds) = 0;
    virtual void reset() = 0;
};

// ---------------------------
// GeneticAlgorithm
// ---------------------------

class GeneticAlgorithm {
public:
    struct StopCriteria {
        size_t max_generations = 200;
        size_t max_stagnant_generations = 50;
        double min_delta = 1e-12;
        double target_fitness = numeric_limits<double>::quiet_NaN();
    };

    struct Config {
        size_t population_size = 50;
        uint32_t seed = 123;
        size_t elitism_count = 1;
        double crossover_rate = 0.9;
        double mutation_rate = 0.2;

        bool keep_history = false;
        StopCriteria stop;
    };

    struct GenerationStats {
        size_t generation = 0;
        double best_fitness = 0.0;
        double mean_fitness = 0.0;
        double worst_fitness = 0.0;
    };

    struct Result {
        Individual best;
        size_t generations_done = 0;

        vector<GenerationStats> history;

        bool stopped_by_max_generations = false;
        bool stopped_by_stagnation = false;
        bool reached_target = false;

        string stop_reason;
    };

    using GenerationCallback = function<void(const GeneticAlgorithm&, const GenerationStats&)>;

public:
    GeneticAlgorithm();
    GeneticAlgorithm(const Config& cfg);

    void setConfig(const Config& cfg);
    const Config& getConfig() const;

    void setSeed(uint32_t seed);
    uint32_t getSeed() const;             // unit32_t int sin signo de 32 bits con tamaño fijo, por consistencia

    // Random helpers
    int randomInt(int lo, int hi) const;                 // seleccion, puntos cruce y individuos
    double randomReal(double lo, double hi) const;       // cruces en BLX y inicializacion de genes reales
    double random01() const;                             // entre [0, 1] prob, decision, activar mut o cruces

    // Problem / Operators injection
    void setProblem(shared_ptr<IProblem> problem);
    shared_ptr<IProblem> getProblem() const;

    void setSelection(unique_ptr<ISelection> selection);
    void setCrossover(unique_ptr<ICrossover> crossover);
    void setMutation(unique_ptr<IMutation> mutation);

    ISelection* getSelection() const;
    ICrossover* getCrossover() const;
    IMutation* getMutation() const;

    void setGenerationCallback(GenerationCallback cb);
    GenerationCallback getGenerationCallback() const;

    // State getters
    size_t getGeneration() const;
    Objective getObjective() const;
    const vector<Bounds>& getBounds() const;
    const vector<Individual>& getPopulation() const;
    const Individual& getBestIndividual() const;
    const vector<GenerationStats>& getHistory() const;

    // Control
    void reset();
    void initialize();
    bool step();
    Result run();

private:
    // Private helpers esto me lo ha hecho chatgptino
    void ensureReady() const;

    void evaluatePopulation(vector<Individual>& pop);
    void evaluateIndividual(Individual& ind);

    bool isBetter(double a, double b) const;

    double computeMeanFitness(const vector<Individual>& pop) const;
    void computeBestWorst(const vector<Individual>& pop, Individual& out_best, Individual& out_worst) const;

    void updateStats();
    void updateStagnation();

    void buildNextPopulation();
    void applyElitism();
    void makeOffspring(Individual& out_child);

    bool shouldStop() const;

private:
    Config cfg_;
    mutable mt19937 rng_;

    shared_ptr<IProblem> problem_;
    unique_ptr<ISelection> selection_;
    unique_ptr<ICrossover> crossover_;
    unique_ptr<IMutation> mutation_;

    GenerationCallback on_generation_;

    size_t generation_ = 0;

    vector<Bounds> bounds_;
    vector<Individual> population_;
    vector<Individual> next_population_;

    Individual best_;
    Individual worst_;

    vector<GenerationStats> history_;

    size_t stagnant_generations_ = 0;
    double last_best_fitness_ = 0.0;
};

// ---------------------------
// Selection
// ---------------------------

class TournamentSelection : public ISelection {
public:
    struct Config {
        size_t tournament_size = 3;
    };

    TournamentSelection();
    TournamentSelection(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    size_t selectIndex(const GeneticAlgorithm& ga, const vector<Individual>& population) override;
    void reset() override;

private:
    Config cfg_;
};

class RouletteSelection : public ISelection {
public:
    struct Config {
        double epsilon = 1e-9;
    };

    RouletteSelection();
    RouletteSelection(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    size_t selectIndex(const GeneticAlgorithm& ga, const vector<Individual>& population) override;
    void reset() override;

private:
    Config cfg_;
};

// ---------------------------
// Crossover
// ---------------------------

class OnePointCrossover : public ICrossover {
public:
    struct Config {
        bool allow_endpoints = false;
    };

    OnePointCrossover();
    OnePointCrossover(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    void crossover(const GeneticAlgorithm& ga,
                   const Genome& parent_a,
                   const Genome& parent_b,
                   Genome& out_child_a,
                   Genome& out_child_b) override;

    void reset() override;

private:
    Config cfg_;
};

class UniformCrossover : public ICrossover {
public:
    struct Config {
        double swap_prob = 0.5;
    };

    UniformCrossover();
    UniformCrossover(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    void crossover(const GeneticAlgorithm& ga,
                   const Genome& parent_a,
                   const Genome& parent_b,
                   Genome& out_child_a,
                   Genome& out_child_b) override;

    void reset() override;

private:
    Config cfg_;
};

class BlendCrossover : public ICrossover {
public:
    struct Config {
        double alpha = 0.5;
    };

    BlendCrossover();
    BlendCrossover(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    void crossover(const GeneticAlgorithm& ga,
                   const Genome& parent_a,
                   const Genome& parent_b,
                   Genome& out_child_a,
                   Genome& out_child_b) override;

    void reset() override;

private:
    Config cfg_;
};

// ---------------------------
// Mutation
// ---------------------------

class BitFlipMutation : public IMutation {
public:
    struct Config {
        double flip_prob = 0.01;
    };

    BitFlipMutation();
    BitFlipMutation(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    void mutate(const GeneticAlgorithm& ga, Genome& inout_genome, const vector<Bounds>&) override;
    void reset() override;

private:
    Config cfg_;
};

class GaussianMutation : public IMutation {
public:
    struct Config {
        double sigma = 0.1;
        bool clamp_to_bounds = true;
    };

    GaussianMutation();
    GaussianMutation(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    void mutate(const GeneticAlgorithm& ga, Genome& inout_genome, const vector<Bounds>& bounds) override;
    void reset() override;

private:
    Config cfg_;
};

} // namespace ga
