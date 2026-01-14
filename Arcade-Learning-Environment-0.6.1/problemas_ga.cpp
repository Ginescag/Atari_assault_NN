#include "ga_problems.h"

#include <cmath>
#include <sstream>

using namespace std;

namespace ga {

static double PI_VALUE() {
    return 3.14159265358979323846;
}

static double clampValue(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

//nota: estos coments NO han sido generados por el chati, los he hecho yo a mano
//os resumo lo que  es el problema de la esfera:
/*
1-
tenemos x dimensiones, pueden ser 2,3,10...
tengo un vector de numeros desde x0 hasta xn-1 y quiero que la 
suma de los cuadrados sea lo mas pequeña posible
es decir, f(x)= x0²+ ... +xn-1²
sumas todos los cuadrados
el minimo obviamente es todos los xi con 0
lo que querems es minimizar f(x)
2-
el genoma es la solucion candidata
cada generacion es un numero real dentro de un rango, por ejemplo -5.12 y +5.12
objetivo final:encontrar un vector lo más cercano posible a todos ceros, porque eso hace que la suma de cuadrados sea mínima.

*/
SphereProblem::SphereProblem() {

}

SphereProblem::SphereProblem(const Config& cfg) : cfg_(cfg) {

}
//cambiar config de la esfera, dimensiones y eso
void SphereProblem::setConfig(const Config& cfg) { 
    cfg_ = cfg; 
}
//getter simple
SphereProblem::Config SphereProblem::getConfig() const { 
    return cfg_; 
}
//genero numeros aleatorios igual que el GA, la idea es usar misma semilla y eso
void SphereProblem::attachGA(const GeneticAlgorithm* ga) { 
    ga_ = ga; 
}
//getter
const GeneticAlgorithm* SphereProblem::getAttachedGA() const { 
    return ga_; 
}
//getter del type
GenomeType SphereProblem::genomeType() const { 
    return GenomeType::Real; 
}
//getter de las dimensiones
size_t SphereProblem::genomeSize() const { 
    return cfg_.dimension; 
}
//getter del objetivo
Objective SphereProblem::objective() const { 
    return Objective::Minimize; 
}
//Rellena out_bounds con los límites por gen
//para cada gen [cfg_.min_value, cfg_.max_value]
//se usa para no salirse del rango en mutaciones
//inicializar valores validos
//evitar soluciones que den error
void SphereProblem::getBounds(vector<Bounds>& out_bounds) const {
    out_bounds.clear();
    out_bounds.resize(cfg_.dimension);
    for (size_t i = 0; i < cfg_.dimension; i++) {
        out_bounds[i].lo = cfg_.min_value;
        out_bounds[i].hi = cfg_.max_value;
    }
}
//creo genoma random para inicializar
void SphereProblem::randomGenome(Genome& out_genome) const {
    out_genome.setType(GenomeType::Real);
    out_genome.resizeReal(cfg_.dimension);

    for (size_t i = 0; i < cfg_.dimension; i++) {
        double v = 0.0;
        //saco un numero aleatorio en ese rango
        if (ga_) v = ga_->randomReal(cfg_.min_value, cfg_.max_value);
        //si no tenemos un ga linkeado, muy raro en la practica, generamos rands y ya
        else     v = cfg_.min_value + (cfg_.max_value - cfg_.min_value) * ((double)rand() / RAND_MAX);
        //el gen i del genoma vale v
        out_genome.setReal(i, v);
    }
}
//suma de cuadrados
double SphereProblem::evaluate(const Genome& genome) const {
    if (genome.getType() != GenomeType::Real) {
        return numeric_limits<double>::infinity();
    }

    double sum = 0.0;
    size_t n = genome.reals.size();
    for (size_t i = 0; i < n; i++) {
        double x = genome.reals[i];
        sum += x * x;
    }
    return sum;
}
//como el to string de toda la vida
string SphereProblem::describe(const Genome& genome) const {
    ostringstream oss;
    oss << "Sphere(dim=" << cfg_.dimension << ") ";
    oss << "x=[";
    size_t n = genome.reals.size();
    size_t show = (n < 5 ? n : 5);
    for (size_t i = 0; i < show; i++) {
        if (i) oss << ",";
        oss << genome.reals[i];
    }
    if (n > show) oss << ",...";
    oss << "]";
    return oss.str();
}


//problema loco del coco
//elijo un vector de numeros y se le aplica una funcion fija que es una locura para ponerla aqui
//el objetivo es hacer que esa funcion sea lo minimo posible
// es literalmente lo mismo que la esfera pero se podria decir que la esfera es la version sencilla
//y este problema es interesante por la complejidad de la funcion y porque tiene muchos minimos locales, lo que lo hace mas dificil

RastriginProblem::RastriginProblem() {}

RastriginProblem::RastriginProblem(const Config& cfg) : cfg_(cfg) {}

void RastriginProblem::setConfig(const Config& cfg) { cfg_ = cfg; }
RastriginProblem::Config RastriginProblem::getConfig() const { return cfg_; }

void RastriginProblem::attachGA(const GeneticAlgorithm* ga) { ga_ = ga; }
const GeneticAlgorithm* RastriginProblem::getAttachedGA() const { return ga_; }

GenomeType RastriginProblem::genomeType() const { return GenomeType::Real; }
size_t RastriginProblem::genomeSize() const { return cfg_.dimension; }
Objective RastriginProblem::objective() const { return Objective::Minimize; }
//todo lo anterior es lo mismo que en el de la esfera
//aqui hago un vector usando la dimension como las entradas y en lo ponemos minimo y hi el maximo.
void RastriginProblem::getBounds(vector<Bounds>& out_bounds) const {
    out_bounds.clear();
    out_bounds.resize(cfg_.dimension);
    for (size_t i = 0; i < cfg_.dimension; i++) {
        out_bounds[i].lo = cfg_.min_value;
        out_bounds[i].hi = cfg_.max_value;
    }
}
//definimos a real y tipo de dimension el problema
//para cada i genero  un valor aleatorio v y lo meto dentro del genoma
//(Creo una solucion aleatoria valida para el problema)
void RastriginProblem::randomGenome(Genome& out_genome) const {
    out_genome.setType(GenomeType::Real);
    out_genome.resizeReal(cfg_.dimension);

    for (size_t i = 0; i < cfg_.dimension; i++) {
        double v = 0.0;
        if (ga_) v = ga_->randomReal(cfg_.min_value, cfg_.max_value);
        else     v = cfg_.min_value + (cfg_.max_value - cfg_.min_value) * ((double)rand() / RAND_MAX);
        out_genome.setReal(i, v);
    }
}
//esta es la parte interesante y crucial, lo que mas cambiará problema a problema.

double RastriginProblem::evaluate(const Genome& genome) const {
    if (genome.getType() != GenomeType::Real) {
        return numeric_limits<double>::infinity();
    }
//defino constantes, se usa 10 porque es lo mas tipico en este problema
    const double A = 10.0;
    const double twoPi = 2.0 * PI_VALUE();
    //se aplica la formula infernal
    double sum = A * (double)genome.reals.size();//A*n
    //para cada variable x hago x² -A *cos(2pi*x)
    for (size_t i = 0; i < genome.reals.size(); i++) {
        double x = genome.reals[i];
        sum += (x * x) - A * cos(twoPi * x);
    }
    return sum;
}
//logs
string RastriginProblem::describe(const Genome& genome) const {
    ostringstream oss;
    oss << "Rastrigin(dim=" << cfg_.dimension << ") ";
    oss << "x0=" << (genome.reals.empty() ? 0.0 : genome.reals[0]);
    return oss.str();
}


/*
os explico el problema
es otra funcion clasica de optimizacion
tiene un minimo global claro
suele tener ondulaciones y zona central complicada
zona plana por fuera y agujero hacia el centro
elegimos n, vector de reales
se pone esta pedazo de funcion
f(x)=−a⋅exp(−b⋅sqrt(1/n​∑xi²​)​)−exp(1/n​∑cos(cxi​))+a+e
pongo sqrt porque no se poner raiz cuadrada en visual
no tiene mas, es minimizar otra vez pero mas tocho
solo explico el evaluate, el resto es lo mismo
*/

AckleyProblem::AckleyProblem() {}

AckleyProblem::AckleyProblem(const Config& cfg) : cfg_(cfg) {}

void AckleyProblem::setConfig(const Config& cfg) { cfg_ = cfg; }
AckleyProblem::Config AckleyProblem::getConfig() const { return cfg_; }

void AckleyProblem::attachGA(const GeneticAlgorithm* ga) { ga_ = ga; }
const GeneticAlgorithm* AckleyProblem::getAttachedGA() const { return ga_; }

GenomeType AckleyProblem::genomeType() const { return GenomeType::Real; }
size_t AckleyProblem::genomeSize() const { return cfg_.dimension; }
Objective AckleyProblem::objective() const { return Objective::Minimize; }

void AckleyProblem::getBounds(vector<Bounds>& out_bounds) const {
    out_bounds.clear();
    out_bounds.resize(cfg_.dimension);
    for (size_t i = 0; i < cfg_.dimension; i++) {
        out_bounds[i].lo = cfg_.min_value;
        out_bounds[i].hi = cfg_.max_value;
    }
}

void AckleyProblem::randomGenome(Genome& out_genome) const {
    out_genome.setType(GenomeType::Real);
    out_genome.resizeReal(cfg_.dimension);

    for (size_t i = 0; i < cfg_.dimension; i++) {
        double v = 0.0;
        if (ga_) v = ga_->randomReal(cfg_.min_value, cfg_.max_value);
        else     v = cfg_.min_value + (cfg_.max_value - cfg_.min_value) * ((double)rand() / RAND_MAX);
        out_genome.setReal(i, v);
    }
}

double AckleyProblem::evaluate(const Genome& genome) const {
    if (genome.getType() != GenomeType::Real) {
        return numeric_limits<double>::infinity();
    }

    size_t n = genome.reals.size();
    if (n == 0) return numeric_limits<double>::infinity();

    const double a = 20.0;
    const double b = 0.2;
    const double c = 2.0 * PI_VALUE();

    double sum_sq = 0.0;//esto es el sumatorio de xi al cuadrado, mide que tan lejos estoy de 0
    double sum_cos = 0.0;//sum del coseno

    for (size_t i = 0; i < n; i++) {
        double x = genome.reals[i];
        sum_sq += x * x;//hago los cuadrados
        sum_cos += cos(c * x);//hago coseno de cx, ambos son sumatorios
    }
    /*
    explicacion del chatgpt
    Si estás lejos del origen, sqrt(sum_sq/n) es grande → exp(-b * grande) se hace pequeño → el término se acerca a 0 (pero negativo).
    Si estás cerca de 0, sqrt(...) es pequeño → exp(-b*pequeño) se acerca a 1 → term1 se acerca a -a.
    O sea: este término empuja a que el mínimo esté cerca del origen*/
    double term1 = -a * exp(-b * sqrt(sum_sq / (double)n));//primera parte de la formula
    /*
    sum_cos/n es el promedio del coseno.
    Si estás justo en 0, cos(0)=1, entonces sum_cos/n = 1 y term2 = -e.
    Este término es el que introduce la “rugosidad” (los mínimos locales).*/
    double term2 = -exp(sum_cos / (double)n);//segunda parte
    double result = term1 + term2 + a + exp(1.0);//se juntan y vemos resultado

    return result;
}

string AckleyProblem::describe(const Genome& genome) const {
    ostringstream oss;
    oss << "Ackley(dim=" << cfg_.dimension << ") ";
    oss << "x0=" << (genome.reals.empty() ? 0.0 : genome.reals[0]);
    return oss.str();
}


/*
Este problema es distinto a los otros, por fin cambiamos,es cn valores binarios como la compuerta xor
hay que buscar una regla para que se cumpla la tabla de verdad de una xor
(solo es 1 si son distintos)
cada bit del 0 al 3 (en un genoma de 4 bits)será la salida que proponga para cada 00,01,10,11
tenemos que buscar el genoma que de [0,1,1,0], basicamente es eso
maximizar aciertos
*/
XorProblem::XorProblem() {}

XorProblem::XorProblem(const Config& cfg) : cfg_(cfg) {}

void XorProblem::setConfig(const Config& cfg) { cfg_ = cfg; }
XorProblem::Config XorProblem::getConfig() const { return cfg_; }

void XorProblem::attachGA(const GeneticAlgorithm* ga) { ga_ = ga; }
const GeneticAlgorithm* XorProblem::getAttachedGA() const { return ga_; }

GenomeType XorProblem::genomeType() const { return GenomeType::Binary; }
size_t XorProblem::genomeSize() const { return cfg_.expected.size(); }
Objective XorProblem::objective() const { return Objective::Maximize; }
//el rango es solo 0,1 por eso aqui clear
void XorProblem::getBounds(vector<Bounds>& out_bounds) const {
    out_bounds.clear(); // binario no usa bounds
}

void XorProblem::randomGenome(Genome& out_genome) const {
    out_genome.setType(GenomeType::Binary);
    out_genome.resizeBinary(cfg_.expected.size());

    for (size_t i = 0; i < cfg_.expected.size(); i++) {
        int b = 0;
        if (ga_) b = ga_->randomInt(0, 1);
        else     b = rand() % 2;
        out_genome.setBit(i, b);
    }
}

double XorProblem::evaluate(const Genome& genome) const {
    if (genome.getType() != GenomeType::Binary) {
        return -numeric_limits<double>::infinity();
    }

    size_t n = cfg_.expected.size();
    if (genome.bits.size() < n) return -numeric_limits<double>::infinity();
    //si por lo que sea el tipo o el size esta mal devolvemos -infinito
    int correct = 0;
    //recorro el vector
    for (size_t i = 0; i < n; i++) {
        //saco el bit del genoma g y saco el del esperado, si coinciden, sumo 1
        int g = (genome.bits[i] != 0) ? 1 : 0;
        int e = (cfg_.expected[i] != 0) ? 1 : 0;
        if (g == e) correct++;
    }

    return (double)correct;
}
s//mas logs
string XorProblem::describe(const Genome& genome) const {
    ostringstream oss;
    oss << "XOR bits=[";
    for (size_t i = 0; i < genome.bits.size(); i++) {
        if (i) oss << " ";
        oss << (genome.bits[i] ? 1 : 0);
    }
    oss << "]";
    return oss.str();
}


/*
segun lo que he investigado este problema es basico para los geneticos porque es bastante claro
tenemos un vector de bits 0,1 de longitud n
queremos contar cuantos unos hay, es solo eso
hay que maximizar el numero de unos
*/

OneMaxProblem::OneMaxProblem() {}

OneMaxProblem::OneMaxProblem(const Config& cfg) : cfg_(cfg) {}

void OneMaxProblem::setConfig(const Config& cfg) { cfg_ = cfg; }
OneMaxProblem::Config OneMaxProblem::getConfig() const { return cfg_; }

void OneMaxProblem::attachGA(const GeneticAlgorithm* ga) { ga_ = ga; }
const GeneticAlgorithm* OneMaxProblem::getAttachedGA() const { return ga_; }

GenomeType OneMaxProblem::genomeType() const { return GenomeType::Binary; }
size_t OneMaxProblem::genomeSize() const { return cfg_.n_bits; }
Objective OneMaxProblem::objective() const { return Objective::Maximize; }

void OneMaxProblem::getBounds(vector<Bounds>& out_bounds) const {
    out_bounds.clear(); // binario no usa bounds
}

void OneMaxProblem::randomGenome(Genome& out_genome) const {
    out_genome.setType(GenomeType::Binary);
    out_genome.resizeBinary(cfg_.n_bits);

    for (size_t i = 0; i < cfg_.n_bits; i++) {
        int b = 0;
        if (ga_) b = ga_->randomInt(0, 1);
        else     b = rand() % 2;
        out_genome.setBit(i, b);
    }
}

double OneMaxProblem::evaluate(const Genome& genome) const {
    if (genome.getType() != GenomeType::Binary) {
        return -numeric_limits<double>::infinity();
    }

    int ones = 0;
    //recorro todos los bits y veo cuantos son distintos de 0 y voy sumando 1
    for (size_t i = 0; i < genome.bits.size(); i++) {
        if (genome.bits[i] != 0) ones++;
    }
    //devuelvo como double para que sea la misma salida que otros problemas
    return (double)ones;
}

string OneMaxProblem::describe(const Genome& genome) const {
    ostringstream oss;
    oss << "OneMax(n=" << cfg_.n_bits << ") ones=" << (int)evaluate(genome);
    return oss.str();
}


// ============================================================
// AleAtariProblem (caja negra)
// ============================================================

AleAtariProblem::AleAtariProblem() {}

AleAtariProblem::AleAtariProblem(const Config& cfg) : cfg_(cfg) {}

void AleAtariProblem::setConfig(const Config& cfg) { cfg_ = cfg; }
AleAtariProblem::Config AleAtariProblem::getConfig() const { return cfg_; }

void AleAtariProblem::attachGA(const GeneticAlgorithm* ga) { ga_ = ga; }
const GeneticAlgorithm* AleAtariProblem::getAttachedGA() const { return ga_; }

int AleAtariProblem::getEpisodesPerEval() const { return cfg_.episodes_per_eval; }
int AleAtariProblem::getMaxStepsPerEpisode() const { return cfg_.max_steps_per_episode; }

GenomeType AleAtariProblem::genomeType() const { return cfg_.type; }
size_t AleAtariProblem::genomeSize() const { return cfg_.genome_size; }
Objective AleAtariProblem::objective() const { return cfg_.obj; }

void AleAtariProblem::fillDefaultBoundsIfNeeded(vector<Bounds>& b) const {
    if (!b.empty()) return;

    b.resize(cfg_.genome_size);
    for (size_t i = 0; i < cfg_.genome_size; i++) {
        b[i].lo = -1.0;
        b[i].hi =  1.0;
    }
}

double AleAtariProblem::worstFitness() const {
    if (cfg_.obj == Objective::Maximize) return -numeric_limits<double>::infinity();
    return numeric_limits<double>::infinity();
}

void AleAtariProblem::getBounds(vector<Bounds>& out_bounds) const {
    out_bounds = cfg_.bounds;
    fillDefaultBoundsIfNeeded(out_bounds);
}

void AleAtariProblem::randomGenome(Genome& out_genome) const {
    // Si el usuario ha dado init_fn, se usa eso
    if (cfg_.init_fn) {
        cfg_.init_fn(*this, out_genome);
        return;
    }

    // Si no hay init_fn, hacemos algo básico:
    if (cfg_.type == GenomeType::Binary) {
        out_genome.setType(GenomeType::Binary);
        out_genome.resizeBinary(cfg_.genome_size);

        for (size_t i = 0; i < cfg_.genome_size; i++) {
            int b = 0;
            if (ga_) b = ga_->randomInt(0, 1);
            else     b = rand() % 2;
            out_genome.setBit(i, b);
        }
        return;
    }

    // Real
    vector<Bounds> b = cfg_.bounds;
    fillDefaultBoundsIfNeeded(b);

    out_genome.setType(GenomeType::Real);
    out_genome.resizeReal(cfg_.genome_size);

    for (size_t i = 0; i < cfg_.genome_size; i++) {
        double lo = b[i].lo;
        double hi = b[i].hi;
        double v = 0.0;
        if (ga_) v = ga_->randomReal(lo, hi);
        else     v = lo + (hi - lo) * ((double)rand() / RAND_MAX);
        out_genome.setReal(i, v);
    }
}

double AleAtariProblem::evaluate(const Genome& genome) const {
    if (!cfg_.fitness_fn) return worstFitness();
    return cfg_.fitness_fn(*this, genome);
}

string AleAtariProblem::describe(const Genome& genome) const {
    if (cfg_.describe_fn) return cfg_.describe_fn(*this, genome);

    ostringstream oss;
    oss << "AleAtari(genome_size=" << cfg_.genome_size
        << ", episodes=" << cfg_.episodes_per_eval
        << ", max_steps=" << cfg_.max_steps_per_episode << ")";
    return oss.str();
}
//Traveling Salesman Problem (TSP) con “Random Keys”
/*
he querido buscar un problema mas dificil
TSP =  tienes N ciudades y quieres encontrar el orden de
visita que hace que el recorrido total sea lo más corto posible, volviendo al inicio.
entrada de coordenadas de ciudades o matrices de distancias
salida permutacion u orden de ciudades
queremos minimizar el recorrido
para N ciudades hay N! soluciones posibles
//es complejo porque hay muchos "casi buenos" que confunden
TSP normalmente requiere un genoma de permutación, pero nuestro GA no lo tiene.
Solución clásica: Random Keys (claves aleatorias)
El genoma es un vector de reales del tamaño N: keys[i].
Para decodificar a un tour:
Ordenas las ciudades por su key (de menor a mayor).
Ese orden es la ruta.
Ejemplo (N=4):
keys = [0.20, 0.90, 0.10, 0.50]
orden por key: ciudad2 (0.10), ciudad0 (0.20), ciudad3 (0.50), ciudad1 (0.90)
tour = [2,0,3,1]
*/
TspRandomKeysProblem::TspRandomKeysProblem() {}

TspRandomKeysProblem::TspRandomKeysProblem(const Config& cfg) : cfg_(cfg) {}

void TspRandomKeysProblem::setConfig(const Config& cfg) {
    cfg_ = cfg;
}

TspRandomKeysProblem::Config TspRandomKeysProblem::getConfig() const {
    return cfg_;
}

void TspRandomKeysProblem::attachGA(const GeneticAlgorithm* ga) {
    ga_ = ga;
}

const GeneticAlgorithm* TspRandomKeysProblem::getAttachedGA() const {
    return ga_;
}

GenomeType TspRandomKeysProblem::genomeType() const {
    return GenomeType::Real;
}

size_t TspRandomKeysProblem::genomeSize() const {
    if (cfg_.use_distance_matrix) return cfg_.dist_matrix.size();
    return cfg_.cities.size();
}

Objective TspRandomKeysProblem::objective() const {
    return Objective::Minimize;
}

void TspRandomKeysProblem::getBounds(vector<Bounds>& out_bounds) const {
    out_bounds.clear();
    size_t n = genomeSize();
    out_bounds.resize(n);
    for (size_t i = 0; i < n; i++) {
        out_bounds[i].lo = 0.0;
        out_bounds[i].hi = 1.0;
    }
}

void TspRandomKeysProblem::randomGenome(Genome& out_genome) const {
    size_t n = genomeSize();
    out_genome.setType(GenomeType::Real);
    out_genome.resizeReal(n);

    for (size_t i = 0; i < n; i++) {
        double v = 0.0;
        if (ga_) v = ga_->randomReal(0.0, 1.0);
        else     v = (double)rand() / (double)RAND_MAX;
        out_genome.setReal(i, v);
    }
}
//hasta aqui es lo mismo de antes
/*

*/
double TspRandomKeysProblem::evaluate(const Genome& genome) const {
    if (genome.getType() != GenomeType::Real) {
        return numeric_limits<double>::infinity();
    }

    size_t n = genomeSize();
    if (n == 0) return numeric_limits<double>::infinity();
    if (genome.reals.size() < n) return numeric_limits<double>::infinity();

    vector<int> tour;
    decodeTour(genome, tour);//convierto keys a ordenes de ciudades

    return tourLength(tour);//devuelo la length
}

string TspRandomKeysProblem::describe(const Genome& genome) const {
    ostringstream oss;
    size_t n = genomeSize();
    oss << "TSP(RandomKeys) n=" << (int)n;

    if (genome.getType() == GenomeType::Real && genome.reals.size() >= n && n > 0) {
        vector<int> tour;
        decodeTour(genome, tour);

        oss << " tour=[";
        size_t show = (n < 5 ? n : 5);
        for (size_t i = 0; i < show; i++) {
            if (i) oss << " ";
            oss << tour[i];
        }
        if (n > show) oss << " ...";
        oss << "]";
    }

    return oss.str();
}
//esto son los helpers
/*
esto tiene dos modos, si es matriz o solo vector
*/
double TspRandomKeysProblem::distanceBetween(int a, int b) const {
    //compruebo rangos para que no pete
    if (a < 0 || b < 0) return numeric_limits<double>::infinity();

    if (cfg_.use_distance_matrix) {
        size_t n = cfg_.dist_matrix.size();
        //compruebo rangos para que no pete
        if ((size_t)a >= n || (size_t)b >= n) return numeric_limits<double>::infinity();
        if (cfg_.dist_matrix[a].size() < n) return numeric_limits<double>::infinity();
        return cfg_.dist_matrix[a][b];//devuelvo la distancia entre las dos ciudades
    }
    //compruebo rangos para que no pete
    size_t n = cfg_.cities.size();
    if ((size_t)a >= n || (size_t)b >= n) return numeric_limits<double>::infinity();
    //devuelvo distancia euclídea
    double dx = cfg_.cities[a].x - cfg_.cities[b].x;
    double dy = cfg_.cities[a].y - cfg_.cities[b].y;
    return sqrt(dx * dx + dy * dy);
}
//permuto las ciudades para hacer tour
void TspRandomKeysProblem::decodeTour(const Genome& genome, vector<int>& out_tour) const {
    size_t n = genomeSize();
    out_tour.clear();
    out_tour.resize(n);
    //creo out tour, un vector desde 0 a n-1
    for (size_t i = 0; i < n; i++) out_tour[i] = (int)i;

   
    //ordeno el vector comparando las keys genone.reals[i], es decir, de cada ciudad
    //si reals [2] es menor que reals[5] la ciudad 2 va antes
    //busco el minimo y voy haciendo swaps
    for (size_t i = 0; i < n; i++) {
        //i es la posicion del tour que estoy rellenando
        //en i 0 decido que ciudad va la primera, i1 segunda etc
        size_t best = i;//asumo que la menor de las que quedan esta en i
        double bestKey = genome.reals[out_tour[i]];//la key candidata mejor es la key de la ciudad que ahora mismo está en out_tour[i]
        //busco desde i+1 hasta n-1
        //si hay alguna ciudad mejor actualizo best_key
        for (size_t j = i + 1; j < n; j++) {
            double k = genome.reals[out_tour[j]];
            if (k < bestKey) {
                bestKey = k;
                best = j;
            }
        }
        //aqui ya se cual es la ciudad con menos coste
        //si esa ciudad no estaba en i entonces hago swap
        if (best != i) {
            int tmp = out_tour[i];
            out_tour[i] = out_tour[best];
            out_tour[best] = tmp;
        }
    }
}

double TspRandomKeysProblem::tourLength(const vector<int>& tour) const {
    size_t n = tour.size();
    if (n < 2) return numeric_limits<double>::infinity();

    double total = 0.0;
    //voy sumando la distancia entre cada parte del vector del tour
    for (size_t i = 0; i + 1 < n; i++) {
        total += distanceBetween(tour[i], tour[i + 1]);
    }
    //en el caso de querer hacer un ciclo, es decir volver a la primera ciudad, esta la variable closed tour
    if (cfg_.closed_tour) {
        total += distanceBetween(tour[n - 1], tour[0]);
    }

    return total;
}
} // namespace ga
