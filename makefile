# Configuración del compilador
CXX = g++
CXXFLAGS = -O3

# Librerías y flags de enlazado
# Se incluye el rpath actual para encontrar libale.so y la librería SDL
LDFLAGS = -Wl,-rpath=. -lSDL
LIBS = src/libale.so

# Nombre del ejecutable de salida
TARGET = hm

# Archivos fuente
SRCS = src/minimal_agent.cpp

# Regla por defecto (make all)
all: $(TARGET)

# Regla de compilación
$(TARGET): $(SRCS)
	$(CXX) $(SRCS) $(LIBS) -o $(TARGET) $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)
