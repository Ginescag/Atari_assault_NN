#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <random>

namespace ga {

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
// Genome e Individual
// ---------------------------

struct Genome {
    GenomeType type = GenomeType::Real;

    // Se usa uno u otro según type
    vector<int> bits;
    vector<double> reals;

    Genome(){
        type=GenomeType::Real;
        
    }
    explicit Genome(GenomeType t){
        type=t;
    }

    void clear(){
        bits = vector<int>();
        reals=vector<double>();
    }
    GenomeType getType() const{
        return type;
    }
    void setType(GenomeType t){
        if(t==type){
            return;
        }
        else{
            if(t==GenomeType::Real){
                bits = vector<int>();

            }else{
                reals=vector<double>();
    
            }
        }
        type = t;
    }

    size_t size() const{
        if(type == GenomeType::Real){
            return reals.size();
        }else{
            return bits.size();
        }
    }

    void resizeBinary(size_t n_bits){
        type=GenomeType::Binary;
        reals=vector<double>();//limpio
        bits.resize(n_bits);
    }
    void resizeReal(size_t n_reals){
        type=GenomeType::Real;
        bits=vector<int>();
        reals.resize(n_reals);
    }

    int getBit(size_t i) const{
        if(bits.size()<=i){
            return -1;//si es error devuelvo -1, si luego queremos usar -1 o cambiar logica ponemos otra cosa
        }else{
            return bits[i];
        }
    }
    void setBit(size_t i, int v){
        if(bits.size()<=i){
            return ;
        }else{
            //he decidido normalizar a 0,1 ya que es el binario, si por lo que sea decidimos cambiar logica esto se quita
            if(v>1){
                v=1;
            }else if (v<0){
                v=0;
            }
            bits[i]=v;
        }
    }

    double getReal(size_t i) const{
        if(reals.size()<=i){
            return numeric_limits<double>::quiet_NaN();//esto se lo he preguntado al chatgpt, lo admito, se refleja en la memoria
        }else{
            return reals[i];
        }
    }

    
    void setReal(size_t i, double v){
        if(reals.size()<=i){
            return ;
        }else{
            
            reals[i]=v;
        }
    }
};

struct Individual {
    Genome genome;

    double fitness = 0.0;
    bool fitness_valid = false;//me dice si el fitness es fiable o si esta desactualizado

    Individual(){

        fitness = 0.0;
        fitness_valid = false;
    }
    explicit Individual(const Genome& g){
        genome=g;
        fitness = 0.0;
        fitness_valid = false;
    }
    //Cada vez que el genoma cambia (mutación o cruce), el fitness anterior deja de corresponder 
    //esta función hace que el resto del sistema sepa hay que reevaluar
    void invalidateFitness(){
        fitness_valid = false;
    }
};


//esta clase es el tipo de problema que usaremos y es como la interfaz
class IProblem {
public:
    // Destructor virtual: importante para poder borrar por puntero base
    virtual ~IProblem() = default;

    // Qué tipo de genoma usa este problema (Binary o Real)
    virtual GenomeType genomeType() const = 0;

    virtual size_t genomeSize() const = 0;

    virtual Objective objective() const = 0;//maximizar o minimizar

    // Límites por generacion 
    virtual void getBounds(vector<Bounds>& out_bounds) const = 0;

    // Crea un genoma aleatorio válido para este problema
    virtual void randomGenome(Genome& out_genome) const = 0;

    // Evalúa el genoma y devuelve un fitness
    virtual double evaluate(const Genome& genome) const = 0;

    //esto es para logs y eso
    virtual string describe(const Genome& genome) const = 0;

};



class GeneticAlgorithm; // forward

class ISelection {
public:
    virtual ~ISelection() = default;


    // Devuelve índice del padre seleccionado
    virtual size_t selectIndex(const GeneticAlgorithm& ga,
                               const vector<Individual>& population) = 0;
    //reiniciamos el estado del operador en caso de que guarde algo
    virtual void reset() = 0;
};

class ICrossover {
public:
    virtual ~ICrossover()=default;

    virtual void crossover(const GeneticAlgorithm& ga,
                           const Genome& parent_a,
                           const Genome& parent_b,
                           Genome& out_child_a,
                           Genome& out_child_b) = 0;

    virtual void reset() = 0;
};

class IMutation {
public:
    virtual ~IMutation()=default;

    virtual void mutate(const GeneticAlgorithm& ga,
                        Genome& inout_genome,
                        const vector<Bounds>& bounds) = 0;

    virtual void reset() = 0;
};

// ---------------------------
// Selección: Tournament, Roulette
// ---------------------------

class TournamentSelection : public ISelection {
public:
    struct Config {
        int tournament_size = 3;
    };

    TournamentSelection();
    explicit TournamentSelection(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    size_t selectIndex(const GeneticAlgorithm& ga,
                       const vector<Individual>& population) override;

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
    explicit RouletteSelection(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    size_t selectIndex(const GeneticAlgorithm& ga,
                       const vector<Individual>& population) override;

    void reset() override;

private:
    Config cfg_;
};

// ---------------------------
// Cruce: 1-point, uniform, blend
// ---------------------------

class OnePointCrossover : public ICrossover {
public:
    struct Config {
        bool allow_endpoints = false;
    };

    OnePointCrossover();
    explicit OnePointCrossover(const Config& cfg);

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
    explicit UniformCrossover(const Config& cfg);

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
    explicit BlendCrossover(const Config& cfg);

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
// Mutación: bit-flip, gaussiana
// ---------------------------

class BitFlipMutation : public IMutation {
public:
    struct Config {
        double flip_prob = 0.01;
    };

    BitFlipMutation();
    explicit BitFlipMutation(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    void mutate(const GeneticAlgorithm& ga,
                Genome& inout_genome,
                const vector<Bounds>& bounds) override;

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
    explicit GaussianMutation(const Config& cfg);

    void setConfig(const Config& cfg);
    Config getConfig() const;

    void mutate(const GeneticAlgorithm& ga,
                Genome& inout_genome,
                const vector<Bounds>& bounds) override;

    void reset() override;

private:
    Config cfg_;
};

// ---------------------------
// GA principal
// ---------------------------

class GeneticAlgorithm {
public:
    struct StopConfig {
        size_t max_generations = 200;

        // si objective=Maximize: parar si best >= target_fitness
        // si objective=Minimize: parar si best <= target_fitness
        double target_fitness = numeric_limits<double>::quiet_NaN();

        size_t max_stagnant_generations = 50;
        double min_delta = 1e-12;
    };

    struct Config {
        size_t population_size = 50;

        double crossover_rate = 0.9;
        double mutation_rate  = 0.2;

        size_t elitism_count = 1;

        uint32_t seed = 123;

        bool keep_history = true;
        StopConfig stop;
    };

    struct GenerationStats {
        size_t generation = 0;
        double best_fitness = 0.0;
        double mean_fitness = 0.0;
        double worst_fitness = 0.0;
    };

    struct Result {
        Individual best;
        vector<GenerationStats> history;

        size_t generations_done = 0;
        bool reached_target = false;
        bool stopped_by_stagnation = false;
        bool stopped_by_max_generations = false;

        string stop_reason;
    };

    using GenerationCallback = function<void(const GeneticAlgorithm& ga,
                                             const GenerationStats& stats)>;

public:
    GeneticAlgorithm();
    explicit GeneticAlgorithm(const Config& cfg);

    // Configuración
    void setConfig(const Config& cfg);
    const Config& getConfig() const;

    void setSeed(uint32_t seed);
    uint32_t getSeed() const;

    // RNG del GA (para operadores/problema)
    int randomInt(int lo, int hi) const;
    double randomReal(double lo, double hi) const;
    double random01() const;

    // Problema y operadores
    void setProblem(shared_ptr<IProblem> problem);
    shared_ptr<IProblem> getProblem() const;

    void setSelection(unique_ptr<ISelection> selection);
    void setCrossover(unique_ptr<ICrossover> crossover);
    void setMutation(unique_ptr<IMutation> mutation);

    ISelection* getSelection() const;
    ICrossover* getCrossover() const;
    IMutation* getMutation() const;

    // Callback por generación
    void setGenerationCallback(GenerationCallback cb);
    GenerationCallback getGenerationCallback() const;

    // Estado / consulta
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
    // Internos
    void ensureReady() const;

    void evaluatePopulation(vector<Individual>& pop);
    void evaluateIndividual(Individual& ind);

    bool isBetter(double a, double b) const;

    double computeMeanFitness(const vector<Individual>& pop) const;
    void computeBestWorst(const vector<Individual>& pop,
                          Individual& out_best,
                          Individual& out_worst) const;

    void buildNextPopulation();
    void applyElitism();
    void makeOffspring(Individual& out_child);

    bool shouldStop() const;
    void updateStagnation();
    void updateStats();

private:
    // Config
    Config cfg_;

    // Dependencias
    shared_ptr<IProblem> problem_;
    unique_ptr<ISelection> selection_;
    unique_ptr<ICrossover> crossover_;
    unique_ptr<IMutation> mutation_;

    // Estado
    mutable mt19937 rng_;
    size_t generation_ = 0;

    vector<Bounds> bounds_;

    vector<Individual> population_;
    vector<Individual> next_population_;

    Individual best_;
    Individual worst_;

    size_t stagnant_generations_ = 0;
    double last_best_fitness_ = 0.0;

    vector<GenerationStats> history_;

    GenerationCallback on_generation_;
};

} // namespace ga
