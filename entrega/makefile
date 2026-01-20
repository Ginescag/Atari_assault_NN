# Makefile para Compilar Red Neuronal y Algoritmo Genético
# Compilador y Flags
CXX      = g++
CXXFLAGS = -O3 -std=c++17

# Carpetas
DIR_NN   = MLP
DIR_GA   = GA

# Nombres de los ejecutables
EXEC_NN  = mlp_test
EXEC_GA  = main_ga

# ==========================================
# Reglas Principales
# ==========================================

all: $(EXEC_NN) $(EXEC_GA)
	@echo ">> Compilacion completa. Ejecutables generados en la raiz."

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