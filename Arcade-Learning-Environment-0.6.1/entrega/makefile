# ==========================================
# Makefile para Proyecto ML (Estructura Modular)
# ==========================================

# Compilador y Flags
CXX      = g++
CXXFLAGS = -O3 -std=c++17

# Carpetas (Ajusta si tus carpetas tienen otro nombre)
DIR_NN   = MLP
DIR_GA   = GA

# Nombres de los ejecutables (se generan en la raíz)
EXEC_NN  = mlp_test
EXEC_GA  = main_ga

# ==========================================
# Reglas Principales
# ==========================================

all: $(EXEC_NN) $(EXEC_GA)
	@echo ">> Compilación completa. Ejecutables generados en la raíz."

nn: $(EXEC_NN)
ga: $(EXEC_GA)

# ==========================================
# Compilación Red Neuronal (NN)
# ==========================================
# Dependencias: Ficheros dentro de la carpeta NN
$(EXEC_NN): $(DIR_NN)/MLPtests.cpp $(DIR_NN)/NeuralNetwork.h
	$(CXX) $(CXXFLAGS) $(DIR_NN)/MLPtests.cpp -o $(EXEC_NN)

# ==========================================
# Compilación Algoritmo Genético (GA)
# ==========================================
# Dependencias: Ficheros dentro de la carpeta GA
# Se añade -I$(DIR_GA) para que encuentren sus propios .h si es necesario
GA_SRC = $(DIR_GA)/main_ga.cpp $(DIR_GA)/new_ga.cpp $(DIR_GA)/problemas_ga.cpp

$(EXEC_GA): $(GA_SRC) $(DIR_GA)/problemas_ga.h
	$(CXX) $(CXXFLAGS) $(GA_SRC) -o $(EXEC_GA)

# ==========================================
# Tests Rápidos (Shortcuts)
# ==========================================

run-donut: $(EXEC_NN)
	./$(EXEC_NN) donut 2000

run-parity: $(EXEC_NN)
	./$(EXEC_NN) parity 8 5000

run-ga-sphere: $(EXEC_GA)
	./$(EXEC_GA) sphere --dim 20 --pop 80 --gen 300

# ==========================================
# Limpieza
# ==========================================
clean:
	rm -f $(EXEC_NN) $(EXEC_GA)
	@echo "Limpieza completada."