// new_ga.cpp
#include "new_ga.h"

namespace ga {

// ---------------------------
// Genome
// ---------------------------

Genome::Genome(){
    type=GenomeType::Real;
}

Genome::Genome(GenomeType t){
    type=t;
}

void Genome::clear(){
    bits = vector<int>();
    reals=vector<double>();
}

GenomeType Genome::getType() const{
    return type;
}

void Genome::setType(GenomeType t){
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

size_t Genome::size() const{
    if(type == GenomeType::Real){
        return reals.size();
    }else{
        return bits.size();
    }
}

void Genome::resizeBinary(size_t n_bits){
    type=GenomeType::Binary;
    reals=vector<double>();
    bits.resize(n_bits);
}

void Genome::resizeReal(size_t n_reals){
    type=GenomeType::Real;
    bits=vector<int>();
    reals.resize(n_reals);
}

int Genome::getBit(size_t i) const{
    if(bits.size()<=i){
        return -1;
    }else{
        return bits[i];
    }
}

void Genome::setBit(size_t i, int v){
    if(bits.size()<=i){
        return ;
    }else{
        if(v>1){
            v=1;
        }else if (v<0){
            v=0;
        }
        bits[i]=v;
    }
}

double Genome::getReal(size_t i) const{
    if(reals.size()<=i){
        return numeric_limits<double>::quiet_NaN();
    }else{
        return reals[i];
    }
}

void Genome::setReal(size_t i, double v){
    if(reals.size()<=i){
        return ;
    }else{
        reals[i]=v;
    }
}

// ---------------------------
// Individual
// ---------------------------

Individual::Individual(){
    fitness = 0.0;
    fitness_valid = false;
}

Individual::Individual(const Genome& g){
    genome=g;
    fitness = 0.0;
    fitness_valid = false;
}

//Cada vez que el genoma cambia (mutación o cruce), el fitness anterior deja de corresponder 
//esta función hace que el resto del sistema sepa hay que reevaluar
void Individual::invalidateFitness(){
    fitness_valid = false;
}

// ---------------------------
// GeneticAlgorithm (Parte A)
// ---------------------------

GeneticAlgorithm::GeneticAlgorithm(){
    rng_.seed(cfg_.seed);
}

GeneticAlgorithm::GeneticAlgorithm(const Config& cfg){
    cfg_ = cfg;
    rng_.seed(cfg_.seed);
}

void GeneticAlgorithm::setConfig(const Config& cfg){
    cfg_ = cfg;
    rng_.seed(cfg_.seed);
}

const GeneticAlgorithm::Config& GeneticAlgorithm::getConfig() const{
    return cfg_;
}

void GeneticAlgorithm::setSeed(uint32_t seed){
    cfg_.seed = seed;
    rng_.seed(seed);
}

uint32_t GeneticAlgorithm::getSeed() const{
    return cfg_.seed;
}

int GeneticAlgorithm::randomInt(int lo, int hi) const{
    if (hi < lo) {
        int tmp = lo;
        lo = hi;
        hi = tmp;
    }
    uniform_int_distribution<int> dist(lo, hi);
    return dist(rng_);
}

double GeneticAlgorithm::randomReal(double lo, double hi) const{
    if (hi < lo) {
        double tmp = lo;
        lo = hi;
        hi = tmp;
    }
    uniform_real_distribution<double> dist(lo, hi);
    return dist(rng_);
}

double GeneticAlgorithm::random01() const{
    uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

void GeneticAlgorithm::setProblem(shared_ptr<IProblem> problem){
    problem_ = problem;
}

shared_ptr<IProblem> GeneticAlgorithm::getProblem() const{
    return problem_;
}

void GeneticAlgorithm::setSelection(unique_ptr<ISelection> selection){
    selection_ = move(selection);
}

void GeneticAlgorithm::setCrossover(unique_ptr<ICrossover> crossover){
    crossover_ = move(crossover);
}

void GeneticAlgorithm::setMutation(unique_ptr<IMutation> mutation){
    mutation_ = move(mutation);
}

ISelection* GeneticAlgorithm::getSelection() const{
    return selection_.get();
}

ICrossover* GeneticAlgorithm::getCrossover() const{
    return crossover_.get();
}

IMutation* GeneticAlgorithm::getMutation() const{
    return mutation_.get();
}

void GeneticAlgorithm::setGenerationCallback(GenerationCallback cb){
    on_generation_ = cb;
}

GeneticAlgorithm::GenerationCallback GeneticAlgorithm::getGenerationCallback() const{
    return on_generation_;
}

size_t GeneticAlgorithm::getGeneration() const{
    return generation_;
}

Objective GeneticAlgorithm::getObjective() const{
    if (problem_) return problem_->objective();
    return Objective::Maximize;
}

const vector<Bounds>& GeneticAlgorithm::getBounds() const{
    return bounds_;
}

const vector<Individual>& GeneticAlgorithm::getPopulation() const{
    return population_;
}

const Individual& GeneticAlgorithm::getBestIndividual() const{
    return best_;
}

const vector<GeneticAlgorithm::GenerationStats>& GeneticAlgorithm::getHistory() const{
    return history_;
}

} // namespace ga
