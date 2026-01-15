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

// GeneticAlgorithm

GeneticAlgorithm::GeneticAlgorithm(){
    setSeed(cfg_.seed);
}

GeneticAlgorithm::GeneticAlgorithm(const Config& cfg): cfg_(cfg){
    setSeed(cfg_.seed);
}

void GeneticAlgorithm::setConfig(const Config& cfg){
    cfg_ = cfg;
    setSeed(cfg_.seed);
}

const GeneticAlgorithm::Config& GeneticAlgorithm::getConfig() const{
    return cfg_;
}

void GeneticAlgorithm::setSeed(uint32_t seed){
    cfg_.seed = seed;
    rng_.seed(seed);
}

uint32_t GeneticAlgorithm::getSeed() const{             // unit32_t int sin signo de 32 bits con tamaño fijo, por consistencia
    return cfg_.seed;
}


// Random helpers

int GeneticAlgorithm::randomInt(int lo, int hi) const{                 // seleccion, puntos cruce y individuos
    if (hi < lo) {
        int tmp = lo;
        lo = hi;
        hi = tmp;
    }
    uniform_int_distribution<int> dist(lo, hi);
    return dist(rng_);
}

double GeneticAlgorithm::randomReal(double lo, double hi) const{        // cruces en BLX y inicializacion de genes reales
    if (hi < lo) {
        double tmp = lo;
        lo = hi;
        hi = tmp;
    }
    uniform_real_distribution<double> dist(lo, hi);
    return dist(rng_);
}

double GeneticAlgorithm::random01() const{                              // entre [0, 1] prob, decision, activar mut o cruces
    uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}



// Problem / Operators injection


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



// State getters

size_t GeneticAlgorithm::getGeneration() const{
    return generation_;
}

Objective GeneticAlgorithm::getObjective() const{
    if (!problem_) return Objective::Maximize;
    return problem_->objective();
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


// Control


void GeneticAlgorithm::reset(){
    generation_ = 0;
    population_.clear();
    next_population_.clear();
    bounds_.clear();
    history_.clear();

    stagnant_generations_ = 0;
    last_best_fitness_ = 0.0;

    best_ = Individual();
    worst_ = Individual();

    if (selection_) selection_->reset();
    if (crossover_) crossover_->reset();
    if (mutation_) mutation_->reset();
}

void GeneticAlgorithm::initialize(){
    ensureReady();

    reset();

    // Bounds (para reales, y para clamp en mutación/cruce)
    problem_->getBounds(bounds_);

    // Crear población inicial
    population_.resize(cfg_.population_size);
    for (size_t i = 0; i < cfg_.population_size; ++i) {
        Genome g(problem_->genomeType());
        problem_->randomGenome(g);
        population_[i] = Individual(g);
    }

    // Evaluar
    evaluatePopulation(population_);

    // Stats iniciales
    updateStats();
    last_best_fitness_ = best_.fitness;
    stagnant_generations_ = 0;

    if (cfg_.keep_history) {
        history_.push_back(GenerationStats{generation_, best_.fitness, computeMeanFitness(population_), worst_.fitness});
    }

    if (on_generation_) {
        GenerationStats s;
        s.generation = generation_;
        s.best_fitness = best_.fitness;
        s.mean_fitness = computeMeanFitness(population_);
        s.worst_fitness = worst_.fitness;
        on_generation_(*this, s);
    }
}

bool GeneticAlgorithm::step(){
    ensureReady();

    if (population_.empty()){
        initialize();
        return shouldStop();
    }

    // Crear siguiente población
    buildNextPopulation();

    // Reemplazar población
    population_ = next_population_;
    next_population_.clear();

    // Evaluar
    evaluatePopulation(population_);

    // Siguiente generación
    generation_++;

    // Stats, historia y parada
    updateStats();
    updateStagnation();

    if (cfg_.keep_history){
        GenerationStats s;
        s.generation = generation_;
        s.best_fitness = best_.fitness;
        s.mean_fitness = computeMeanFitness(population_);
        s.worst_fitness = worst_.fitness;
        history_.push_back(s);
    }

    if (on_generation_){
        GenerationStats s;
        s.generation = generation_;
        s.best_fitness = best_.fitness;
        s.mean_fitness = computeMeanFitness(population_);
        s.worst_fitness = worst_.fitness;
        on_generation_(*this, s);
    }

    return shouldStop();
}

GeneticAlgorithm::Result GeneticAlgorithm::run(){
    ensureReady();

    if (population_.empty()){
        initialize();
    }

    while (true){
        if (shouldStop()) break;
        bool stop = step();
        if (stop) break;
    }

    Result r;
    r.best = best_;
    r.generations_done = generation_;

    if (cfg_.keep_history){
        r.history = history_;
    }

    // Razonamiento de parada (simple)
    r.stopped_by_max_generations = (generation_ >= cfg_.stop.max_generations);

    if (!isnan(cfg_.stop.target_fitness)){
        if (getObjective() == Objective::Maximize) {
            r.reached_target = (best_.fitness >= cfg_.stop.target_fitness);
        } else {
            r.reached_target = (best_.fitness <= cfg_.stop.target_fitness);
        }
    }

    r.stopped_by_stagnation = (stagnant_generations_ >= cfg_.stop.max_stagnant_generations);

    if (r.reached_target) r.stop_reason = "target fitness reached";
    else if (r.stopped_by_stagnation) r.stop_reason = "stagnation";
    else if (r.stopped_by_max_generations) r.stop_reason = "max generations";
    else r.stop_reason = "stopped";

    return r;
}


// Private helpers esto me lo ha hecho chatgptino

void GeneticAlgorithm::ensureReady() const{
    if (!problem_) {
        cerr << "GA error: problem not set" << endl;
        exit(-1);
    }
    if (!selection_) {
        cerr << "GA error: selection not set" << endl;
        exit(-1);
    }
    if (!crossover_) {
        cerr << "GA error: crossover not set" << endl;
        exit(-1);
    }
    if (!mutation_) {
        cerr << "GA error: mutation not set" << endl;
        exit(-1);
    }
    if (cfg_.population_size == 0) {
        cerr << "GA error: population_size = 0" << endl;
        exit(-1);
    }
}

void GeneticAlgorithm::evaluatePopulation(vector<Individual>& pop){
    for (size_t i = 0; i < pop.size(); ++i) {
        evaluateIndividual(pop[i]);
    }
}

void GeneticAlgorithm::evaluateIndividual(Individual& ind){
    if (ind.fitness_valid) return;

    ind.fitness = problem_->evaluate(ind.genome);
    ind.fitness_valid = true;
}

bool GeneticAlgorithm::isBetter(double a, double b) const{
    if (getObjective() == Objective::Maximize) {
        return a > b;
    } else {
        return a < b;
    }
}

double GeneticAlgorithm::computeMeanFitness(const vector<Individual>& pop) const{
    if (pop.empty()) return 0.0;

    double sum = 0.0;
    for (size_t i = 0; i < pop.size(); ++i) {
        sum += pop[i].fitness;
    }
    return sum / static_cast<double>(pop.size());
}

void GeneticAlgorithm::computeBestWorst(const vector<Individual>& pop, Individual& out_best, Individual& out_worst) const{
    if (pop.empty()) return;

    out_best = pop[0];
    out_worst = pop[0];

    for (size_t i = 1; i < pop.size(); ++i) {
        if (isBetter(pop[i].fitness, out_best.fitness)) {
            out_best = pop[i];
        }
        if (isBetter(out_worst.fitness, pop[i].fitness)) {
            out_worst = pop[i];
        }
    }
}

void GeneticAlgorithm::updateStats(){
    if (population_.empty()) return;
    computeBestWorst(population_, best_, worst_);
}

void GeneticAlgorithm::updateStagnation(){
    // Si no hay referencia previa, inicializamos
    if (generation_ == 0){
        last_best_fitness_ = best_.fitness;
        stagnant_generations_ = 0;
        return;
    }

    double delta = fabs(best_.fitness - last_best_fitness_);

    if (delta <= cfg_.stop.min_delta){
        stagnant_generations_++;
    } else{
        stagnant_generations_ = 0;
        last_best_fitness_ = best_.fitness;
    }
}

void GeneticAlgorithm::buildNextPopulation(){
    next_population_.clear();
    next_population_.resize(cfg_.population_size);

    // 1. elitismo (copiar mejores)
    // 2. resto por reproducción
    // se hace primero elitismo y luego se ompleta

    // Clonar población para poder ordenar facil
    vector<Individual> sorted = population_;

    for (size_t i = 0; i < sorted.size(); ++i){
        size_t best_pos = i;
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (isBetter(sorted[j].fitness, sorted[best_pos].fitness)){
                best_pos = j;
            }
        }
        if (best_pos != i) {
            Individual tmp = sorted[i];
            sorted[i] = sorted[best_pos];
            sorted[best_pos] = tmp;
        }
    }

    // Copiar elites
    size_t elite_count = cfg_.elitism_count;
    if (elite_count > cfg_.population_size) elite_count = cfg_.population_size;

    for (size_t i = 0; i < elite_count; ++i){
        next_population_[i] = sorted[i];
    }

    // Rellenar el resto
    for (size_t i = elite_count; i < cfg_.population_size; ++i){
        makeOffspring(next_population_[i]);
    }
}

void GeneticAlgorithm::applyElitism(){
    // No se usa, se usa en buildNextPopulation
}

void GeneticAlgorithm::makeOffspring(Individual& out_child){
    // Seleccionar padres
    size_t ia = selection_->selectIndex(*this, population_);
    size_t ib = selection_->selectIndex(*this, population_);

    const Individual& pa = population_[ia];
    const Individual& pb = population_[ib];

    Genome child_a(pa.genome.getType());
    Genome child_b(pa.genome.getType());

    // Cruce con probabilidad cfg_.crossover_rate
    if (random01() < cfg_.crossover_rate){
        crossover_->crossover(*this, pa.genome, pb.genome, child_a, child_b);
    } else{
        child_a = pa.genome;
        child_b = pb.genome;
    }

    // Elegir uno de los dos hijos al azar
    Genome chosen = (random01() < 0.5) ? child_a : child_b;

    // Mutación con probabilidad cfg_.mutation_rate
    if (random01() < cfg_.mutation_rate){
        mutation_->mutate(*this, chosen, bounds_);
    }

    out_child = Individual(chosen);
    out_child.invalidateFitness();
}

bool GeneticAlgorithm::shouldStop() const{
    // max generaciones
    if (generation_ >= cfg_.stop.max_generations) {
        return true;
    }

    // target fitness
    if (!isnan(cfg_.stop.target_fitness)){
        if (getObjective() == Objective::Maximize) {
            if (best_.fitness >= cfg_.stop.target_fitness) return true;
        } else {
            if (best_.fitness <= cfg_.stop.target_fitness) return true;
        }
    }

    // estancamiento
    if (stagnant_generations_ >= cfg_.stop.max_stagnant_generations){
        return true;
    }

    return false;
}

// ---------------------------
// Selection
// ---------------------------

TournamentSelection::TournamentSelection(){
    // Confg por defecto (tournament_size = 3)
}

TournamentSelection::TournamentSelection(const Config& cfg): cfg_(cfg){
}

void TournamentSelection::setConfig(const Config& cfg){
    cfg_ = cfg;
}

TournamentSelection::Config TournamentSelection::getConfig() const{
    return cfg_;
}

size_t TournamentSelection::selectIndex(const GeneticAlgorithm& ga, const vector<Individual>& population){
    // Asumimos que la población no está vacía
    size_t pop_size = population.size();

    // Si la población es más grande que el tournament la ajustamos
    size_t t_size = cfg_.tournament_size;
    if (t_size > pop_size) {
        t_size = pop_size;
    }

    // Primer candidato al azar
    size_t best_index = ga.randomInt(0, static_cast<int>(pop_size - 1));
    double best_fitness = population[best_index].fitness;

    // Resto del torneo
    for (size_t i = 1; i < t_size; ++i){
        size_t candidate_index = ga.randomInt(0, static_cast<int>(pop_size - 1));
        double candidate_fitness = population[candidate_index].fitness;

        if (ga.getObjective() == Objective::Maximize){
            if (candidate_fitness > best_fitness){
                best_fitness = candidate_fitness;
                best_index = candidate_index;
            }
        } else{ // Objective::Minimize
            if (candidate_fitness < best_fitness){
                best_fitness = candidate_fitness;
                best_index = candidate_index;
            }
        }
    }

    return best_index;
}

void TournamentSelection::reset()
{
    // No hay estado interno que reiniciar
}


// RouletteSelection
// Cada individuo tiene una prob proporcional a su fitness (No determinista)
// Más suave que torneo
// Para minimizar lo transformamos

RouletteSelection::RouletteSelection(){
    // Cofig por defecto (epsilon = 1e-9). El epsilon es para evitar problemas númericos en la seleccion de la rule, que los pesos>0
}

RouletteSelection::RouletteSelection(const Config& cfg): cfg_(cfg){
}

void RouletteSelection::setConfig(const Config& cfg){
    cfg_ = cfg;
}

RouletteSelection::Config RouletteSelection::getConfig() const{
    return cfg_;
}

size_t RouletteSelection::selectIndex(const GeneticAlgorithm& ga, const vector<Individual>& population){
    if (population.empty()) return 0;

    const size_t pop_size = population.size();

    // 1. Encontrar el mínimo fitness para desplazar y evitar pesos negativos
    double min_fit = population[0].fitness;
    for (size_t i = 1; i < pop_size; ++i) {
        if (population[i].fitness < min_fit){
            min_fit = population[i].fitness;
        }
    }

    // 2. Calcular pesos
    vector<double> weights(pop_size, 0.0);
    double total_weight = 0.0;

    if (ga.getObjective() == Objective::Maximize){
        // Si peso grande mejor (fitness grande)
        // Para hacerlo positivo (fit - min_fit + epsilon)
        for (size_t i = 0; i < pop_size; ++i){
            double w = (population[i].fitness - min_fit) + cfg_.epsilon;
            if (w < cfg_.epsilon) w = cfg_.epsilon; // por si acaso
            weights[i] = w;
            total_weight += w;
        }
    } else{
        // Minimize: peso grande = mejor (fitness pequeño)
        // Lo transformamos 1 / (fit - min_fit + epsilon)
        for (size_t i = 0; i < pop_size; ++i){
            double denom = (population[i].fitness - min_fit) + cfg_.epsilon;
            if (denom < cfg_.epsilon) denom = cfg_.epsilon; // para que no divida por 0
            double w = 1.0 / denom;
            weights[i] = w;
            total_weight += w;
        }
    }

    // Por seguridad, si por la cara total_weight no es usable, devolvemos uno al azar
    if (total_weight <= 0.0) {
        return static_cast<size_t>(ga.randomInt(0, static_cast<int>(pop_size - 1)));
    }

    // 3. Ruleta: elegir un punto aleatorio en [0, total_weight)
    double r = ga.randomReal(0.0, total_weight);

    // 4. Hasta superar r
    double acc = 0.0;
    for (size_t i = 0; i < pop_size; ++i) {
        acc += weights[i];
        if (acc >= r) {
            return i;
        }
    }

    // Por seguridad numérica, devolver el último
    return pop_size - 1;
}

void RouletteSelection::reset(){
    // No hay estado interno que reiniciar
}


// Cruce

// OnePointCrossover
// Se elige un punto de corte
// Hijo A: primeros genes del padre A, resto de B
// Hijo B: primeros genes del padre B, resto de A
// Binary/Real

OnePointCrossover::OnePointCrossover(){
    // allow_endpoints = false por defecto
}

OnePointCrossover::OnePointCrossover(const Config& cfg): cfg_(cfg){
}

void OnePointCrossover::setConfig(const Config& cfg){
    cfg_ = cfg;
}

OnePointCrossover::Config OnePointCrossover::getConfig() const{
    return cfg_;
}

void OnePointCrossover::crossover(const GeneticAlgorithm& ga,const Genome& parent_a,const Genome& parent_b,Genome& out_child_a,Genome& out_child_b){
    // Si los tipos no coinciden copiar sin cruzar
    if (parent_a.getType() != parent_b.getType()){
        out_child_a = parent_a;
        out_child_b = parent_b;
        return;
    }

    GenomeType t = parent_a.getType();
    size_t n = parent_a.size();

    // Si el tamaño no coincide o no hay suficientes genes, copiamos
    if (n == 0 || n != parent_b.size() || n < 2){
        out_child_a = parent_a;
        out_child_b = parent_b;
        return;
    }

    // Elegir punto de corte
    // - sin endpoints: [1, n-1]
    // - con endpoints: [0, n]
    size_t cut = 0;

    if (cfg_.allow_endpoints){
        cut = static_cast<size_t>(ga.randomInt(0, static_cast<int>(n)));
    } else{
        cut = static_cast<size_t>(ga.randomInt(1, static_cast<int>(n - 1)));
    }

    // Preparar hijos con tipo y tamaño
    out_child_a.setType(t);
    out_child_b.setType(t);

    if (t == GenomeType::Binary){
        out_child_a.resizeBinary(n);
        out_child_b.resizeBinary(n);

        for (size_t i = 0; i < n; ++i){
            if (i < cut){
                out_child_a.setBit(i, parent_a.getBit(i));
                out_child_b.setBit(i, parent_b.getBit(i));
            } else{
                out_child_a.setBit(i, parent_b.getBit(i));
                out_child_b.setBit(i, parent_a.getBit(i));
            }
        }
    } else{ // GenomeType::Real
        out_child_a.resizeReal(n);
        out_child_b.resizeReal(n);

        for (size_t i = 0; i < n; ++i) {
            if (i < cut) {
                out_child_a.setReal(i, parent_a.getReal(i));
                out_child_b.setReal(i, parent_b.getReal(i));
            } else {
                out_child_a.setReal(i, parent_b.getReal(i));
                out_child_b.setReal(i, parent_a.getReal(i));
            }
        }
    }
}

void OnePointCrossover::reset(){
    // No hay estado interno que reiniciar
}



// UniformCrossover
// Para cada gen, con prob swap_prob:
// Hijo A coge gen de Padre B
// HIjo B coge gen de Padre A
// Sino se quedan como estan


UniformCrossover::UniformCrossover(){
    // swap_prob = 0.5 por defecto
}

UniformCrossover::UniformCrossover(const Config& cfg): cfg_(cfg){
}

void UniformCrossover::setConfig(const Config& cfg){
    cfg_ = cfg;
}

UniformCrossover::Config UniformCrossover::getConfig() const{
    return cfg_;
}

void UniformCrossover::crossover(const GeneticAlgorithm& ga,const Genome& parent_a, const Genome& parent_b, Genome& out_child_a, Genome& out_child_b){
    // Si los tipos no coinciden, copiamos sin cruzar
    if (parent_a.getType() != parent_b.getType()){
        out_child_a = parent_a;
        out_child_b = parent_b;
        return;
    }

    GenomeType t = parent_a.getType();
    size_t n = parent_a.size();

    // Si tamaños no válidos o distintos, copiamos
    if (n == 0 || n != parent_b.size()){
        out_child_a = parent_a;
        out_child_b = parent_b;
        return;
    }

    // Preparar hijos
    out_child_a.setType(t);
    out_child_b.setType(t);

    // Asegurar swap_prob en [0,1]
    double p = cfg_.swap_prob;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;

    if (t == GenomeType::Binary){
        out_child_a.resizeBinary(n);
        out_child_b.resizeBinary(n);

        for (size_t i = 0; i < n; ++i){
            // Defualt copia directa
            int a = parent_a.getBit(i);
            int b = parent_b.getBit(i);

            // Con probabilidad p, intercambiamos
            if (ga.random01() < p) {
                out_child_a.setBit(i, b);
                out_child_b.setBit(i, a);
            } else {
                out_child_a.setBit(i, a);
                out_child_b.setBit(i, b);
            }
        }
    } else{ // GenomeType::Real
        out_child_a.resizeReal(n);
        out_child_b.resizeReal(n);

        for (size_t i = 0; i < n; ++i){
            double a = parent_a.getReal(i);
            double b = parent_b.getReal(i);

            if (ga.random01() < p){
                out_child_a.setReal(i, b);
                out_child_b.setReal(i, a);
            } else{
                out_child_a.setReal(i, a);
                out_child_b.setReal(i, b);
            }
        }
    }
}

void UniformCrossover::reset(){
    // No hay estado interno que reiniciar
}




// BlendCrossover
// Para cada gen se toma el intervalo entre los dos padres y se amplía con aplha.
// Se elige un valor aleatorio de ese int.
// Si hay bounds en el GA se recoorta
// Se usa para el type Real,


BlendCrossover::BlendCrossover(){
    // alpha = 0.5 por defecto
}

BlendCrossover::BlendCrossover(const Config& cfg): cfg_(cfg){
}

void BlendCrossover::setConfig(const Config& cfg){
    cfg_ = cfg;
}

BlendCrossover::Config BlendCrossover::getConfig() const{
    return cfg_;
}

void BlendCrossover::crossover(const GeneticAlgorithm& ga, const Genome& parent_a, const Genome& parent_b,Genome& out_child_a, Genome& out_child_b){
    // Si tipos distintos, copiamos
    if (parent_a.getType() != parent_b.getType()){
        out_child_a = parent_a;
        out_child_b = parent_b;
        return;
    }

    GenomeType t = parent_a.getType();
    size_t n = parent_a.size();

    // Si tamaños no válidos o distintos, copiamos
    if (n == 0 || n != parent_b.size()){
        out_child_a = parent_a;
        out_child_b = parent_b;
        return;
    }

    // Este cruce está pensado para reales
    if (t != GenomeType::Real) {
        out_child_a = parent_a;
        out_child_b = parent_b;
        return;
    }

    // Preparar hijos
    out_child_a.setType(GenomeType::Real);
    out_child_b.setType(GenomeType::Real);
    out_child_a.resizeReal(n);
    out_child_b.resizeReal(n);

    // alpha no negativo
    double a = cfg_.alpha;
    if (a < 0.0) a = 0.0;

    // Bounds del GA (puede venir vacío)
    const vector<Bounds>& bounds = ga.getBounds();
    bool has_bounds = (bounds.size() == n);

    for (size_t i = 0; i < n; ++i){
        double x1 = parent_a.getReal(i);
        double x2 = parent_b.getReal(i);

        double lo = (x1 < x2) ? x1 : x2;
        double hi = (x1 < x2) ? x2 : x1;

        double d = hi - lo;

        // Intervalo ampliado
        double minv = lo - a * d;
        double maxv = hi + a * d;

        // Generar dos hijos independientes dentro del rango
        double c1 = ga.randomReal(minv, maxv);
        double c2 = ga.randomReal(minv, maxv);

        // Recortar a bounds si existen
        if (has_bounds){
            if (c1 < bounds[i].lo) c1 = bounds[i].lo;
            if (c1 > bounds[i].hi) c1 = bounds[i].hi;

            if (c2 < bounds[i].lo) c2 = bounds[i].lo;
            if (c2 > bounds[i].hi) c2 = bounds[i].hi;
        }

        out_child_a.setReal(i, c1);
        out_child_b.setReal(i, c2);
    }
}

void BlendCrossover::reset(){
    // No hay estado interno que reiniciar
}



// Mutaciones

// BitFlipMutation
// Para cada bit:
// Con prob flip_prob
// Cambia el bit

BitFlipMutation::BitFlipMutation(){
    // flip_prob = 0.01 por defecto
}

BitFlipMutation::BitFlipMutation(const Config& cfg): cfg_(cfg){
}

void BitFlipMutation::setConfig(const Config& cfg){
    cfg_ = cfg;
}

BitFlipMutation::Config BitFlipMutation::getConfig() const{
    return cfg_;
}

void BitFlipMutation::mutate(const GeneticAlgorithm& ga, Genome& inout_genome, const vector<Bounds>&){
    // Solo se hace en Binarios
    if (inout_genome.getType() != GenomeType::Binary){
        return;
    }

    size_t n = inout_genome.size();
    if (n == 0){
        return;
    }

    // Asegurar flip_prob en [0,1]
    double p = cfg_.flip_prob;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;

    for (size_t i = 0; i < n; ++i){
        if (ga.random01() < p) {
            int bit = inout_genome.getBit(i);
            // Flip: 0 -> 1, 1 -> 0
            inout_genome.setBit(i, bit == 0 ? 1 : 0);
        }
    }
}

void BitFlipMutation::reset(){
    // No hay estado interno que reiniciar
}




// GaussianMutation
// Recorre cada gen x
// Con prob: x = x + N(0, sigma)
// sigma controla cuanto se mueve (pequeño = cambios suave, grande = cambios agresivos)


GaussianMutation::GaussianMutation(){
    // sigma = 0.1 por defecto, clamp_to_bounds = true
}

GaussianMutation::GaussianMutation(const Config& cfg): cfg_(cfg){
}

void GaussianMutation::setConfig(const Config& cfg){
    cfg_ = cfg;
}

GaussianMutation::Config GaussianMutation::getConfig() const{
    return cfg_;
}

// Genera un número con distribución normal N(0,1) usando Box-Muller.
// (No es un método del .h, lo dejamos como helper)
static double gaussian01(const GeneticAlgorithm& ga){
    // Evitar log(0)
    double u1 = ga.random01();
    if (u1 < 1e-12) u1 = 1e-12;

    double u2 = ga.random01();

    double r = sqrt(-2.0 * log(u1));
    double theta = 6.283185307179586 * u2; // 2*pi

    return r * cos(theta); // N(0,1)
}

void GaussianMutation::mutate(const GeneticAlgorithm& ga, Genome& inout_genome, const vector<Bounds>& bounds){
    // Solo tiene sentido para genomas reales
    if (inout_genome.getType() != GenomeType::Real){
        return;
    }

    size_t n = inout_genome.size();
    if (n == 0){
        return;
    }

    // sigma no negativa
    double sigma = cfg_.sigma;
    if (sigma < 0.0) sigma = 0.0;

    bool has_bounds = (bounds.size() == n);

    for (size_t i = 0; i < n; ++i){
        double x = inout_genome.getReal(i);

        // Mutación gaussiana: x = x + sigma * N(0,1)           // Box–Muller es un método que transforma números aleatorios uniformes en números
        double noise = gaussian01(ga) * sigma;                  // con distribución normal, y lo uso para generar el ruido gaussiano de la mutación.
        x += noise;

        // Recortar si hay bounds y se pide clamp
        if (cfg_.clamp_to_bounds && has_bounds){
            if (x < bounds[i].lo) x = bounds[i].lo;
            if (x > bounds[i].hi) x = bounds[i].hi;
        }

        inout_genome.setReal(i, x);
    }
}

void GaussianMutation::reset(){
    // No hay estado interno que reiniciar
}

} // namespace ga
