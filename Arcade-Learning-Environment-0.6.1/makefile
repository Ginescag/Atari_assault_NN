# =========================
# ALE + Proyecto (Makefile)
# =========================

# ---- Configuración básica ----
CXX       ?= g++
CC        ?= gcc
AR        ?= ar
RM        ?= rm -f

BUILD_DIR ?= build
ROM       ?= supported/assault.bin

# 1 => compila ALE con SDL (display + sound) y compila tu minimal_agent con SDL
# 0 => modo headless (sin SDL). Ojo: minimal_agent incluye SDL/SDL.h, así que sin SDL no compilará.
USE_SDL   ?= 1

# ---- Flags (alineado con CMakeLists.txt de ALE) ----
COMMON_DEFS := -DHAVE_INTTYPES -D__STDC_CONSTANT_MACROS
OPTFLAGS     := -O3 -fomit-frame-pointer
WARNFLAGS    := -Wall -Wextra -Wshadow -Wconversion -Wno-unused-parameter -Wno-unused-function
PICFLAGS     := -fPIC

# Dependencias automáticas .d
DEPFLAGS     := -MMD -MP

CXXSTD       := -std=c++11

CFLAGS   := $(OPTFLAGS) $(WARNFLAGS) $(PICFLAGS) $(DEPFLAGS) $(COMMON_DEFS)
CXXFLAGS := $(CXXSTD) $(OPTFLAGS) $(WARNFLAGS) $(PICFLAGS) $(DEPFLAGS) $(COMMON_DEFS)

# ---- Includes de ALE (como en CMakeLists) ----
ALE_SRC_DIR := src
ALE_INCLUDES := \
  -I$(ALE_SRC_DIR) \
  -I$(ALE_SRC_DIR)/common \
  -I$(ALE_SRC_DIR)/controllers \
  -I$(ALE_SRC_DIR)/emucore \
  -I$(ALE_SRC_DIR)/emucore/m6502/src \
  -I$(ALE_SRC_DIR)/emucore/m6502/src/bspf/src \
  -I$(ALE_SRC_DIR)/environment \
  -I$(ALE_SRC_DIR)/games \
  -I$(ALE_SRC_DIR)/games/supported \
  -I$(ALE_SRC_DIR)/os_dependent \
  -I$(ALE_SRC_DIR)/external \
  -I$(ALE_SRC_DIR)/external/TinyMT

CFLAGS   += $(ALE_INCLUDES)
CXXFLAGS += $(ALE_INCLUDES)

# ---- Link libs base ----
LDLIBS := -lz

# ---- SDL (solo si USE_SDL=1) ----
ifeq ($(USE_SDL),1)
  CFLAGS   += -D__USE_SDL -DSOUND_SUPPORT
  CXXFLAGS += -D__USE_SDL -DSOUND_SUPPORT

  # Preferimos sdl-config (SDL 1.2). Si no existe, intentamos pkg-config.
  SDL_CFLAGS ?= $(shell sdl-config --cflags 2>/dev/null || pkg-config --cflags sdl 2>/dev/null)
  SDL_LIBS   ?= $(shell sdl-config --libs   2>/dev/null || pkg-config --libs sdl   2>/dev/null)

  CFLAGS   += $(SDL_CFLAGS)
  CXXFLAGS += $(SDL_CFLAGS)
  LDLIBS   += $(SDL_LIBS)
endif

# ---- Detectar OS para fuentes os_dependent (asumimos UNIX si no es Windows) ----
UNAME_S := $(shell uname -s 2>/dev/null)
ifeq ($(UNAME_S),)
  UNAME_S := Unknown
endif

# En ALE original (CMake) para UNIX:
ALE_OS_SRCS_UNIX := \
  $(ALE_SRC_DIR)/os_dependent/SettingsUNIX.cxx \
  $(ALE_SRC_DIR)/os_dependent/OSystemUNIX.cxx \
  $(ALE_SRC_DIR)/os_dependent/FSNodePOSIX.cxx

# (si algún día compilas en Windows, tendrás que ajustar estas rutas)
ALE_OS_SRCS_WIN := \
  $(ALE_SRC_DIR)/os_dependent/SettingsWin32.cxx \
  $(ALE_SRC_DIR)/os_dependent/OSystemWin32.cxx \
  $(ALE_SRC_DIR)/os_dependent/FSNodeWin32.cxx

# Selección simple: si parece windows -> WIN, si no -> UNIX
ifeq ($(findstring MINGW,$(UNAME_S)),MINGW)
  ALE_OS_SRCS := $(ALE_OS_SRCS_WIN)
else ifeq ($(findstring MSYS,$(UNAME_S)),MSYS)
  ALE_OS_SRCS := $(ALE_OS_SRCS_WIN)
else ifeq ($(findstring CYGWIN,$(UNAME_S)),CYGWIN)
  ALE_OS_SRCS := $(ALE_OS_SRCS_WIN)
else
  ALE_OS_SRCS := $(ALE_OS_SRCS_UNIX)
endif

# ---- Fuentes ALE (mismo listado de módulos que CMakeLists) ----
ALE_MODULES := \
  common controllers emucore emucore/m6502/src emucore/m6502/src/bspf/src \
  environment games games/supported external external/TinyMT

# CMake hacía GLOB en cada módulo para *.c y *.c?[xp] (cc/cpp/cxx)
ALE_MODULE_SRCS := $(foreach m,$(ALE_MODULES), \
  $(wildcard $(ALE_SRC_DIR)/$(m)/*.c)   \
  $(wildcard $(ALE_SRC_DIR)/$(m)/*.cc)  \
  $(wildcard $(ALE_SRC_DIR)/$(m)/*.cpp) \
  $(wildcard $(ALE_SRC_DIR)/$(m)/*.cxx) \
)

# Core ALE
ALE_CORE_SRCS := $(ALE_SRC_DIR)/ale_interface.cpp $(ALE_MODULE_SRCS) $(ALE_OS_SRCS)

# CLI ale (como CMake: src/main.cpp + ale_interface.cpp + resto)
ALE_CLI_MAIN := $(firstword $(wildcard $(ALE_SRC_DIR)/main.cpp))
ALE_CLI_SRCS := $(ALE_CLI_MAIN) $(ALE_CORE_SRCS)

# ---- Helpers: map src -> obj en build/ ----
define to_obj
$(BUILD_DIR)/$(basename $(1)).o
endef

ALE_CORE_OBJS := $(foreach s,$(ALE_CORE_SRCS),$(call to_obj,$(s)))
ALE_CLI_OBJS  := $(foreach s,$(ALE_CLI_SRCS),$(call to_obj,$(s)))

# ---- Outputs ----
ALE_SO   := libale.so
ALE_BIN  := ale

# ---- Apps del proyecto (solo si existen los .cpp) ----
HAS_MINIMAL := $(wildcard minimal_agent.cpp)
HAS_RCGA    := $(wildcard rcga_train_ale.cpp)

ifeq ($(HAS_MINIMAL),)
  MINIMAL_BIN :=
  MINIMAL_OBJS :=
else
  MINIMAL_BIN  := minimal_agent
  MINIMAL_OBJS := $(call to_obj,minimal_agent.cpp)
endif

ifeq ($(HAS_RCGA),)
  RCGA_BIN :=
  RCGA_OBJS :=
else
  RCGA_BIN  := rcga_train_ale
  RCGA_OBJS := $(call to_obj,rcga_train_ale.cpp)
endif

# ---- Python C wrapper (si existe) ----
ALE_C_WRAPPER := $(firstword $(wildcard ale_python_interface/ale_c_wrapper.cpp ale_c_wrapper.cpp))
ifeq ($(ALE_C_WRAPPER),)
  ALE_C_SO :=
  ALE_C_OBJS :=
else
  ALE_C_SO   := ale_python_interface/libale_c.so
  ALE_C_OBJS := $(call to_obj,$(ALE_C_WRAPPER)) $(ALE_CORE_OBJS)
endif

# ---- Dependencias (.d) ----
ALL_OBJS := $(ALE_CORE_OBJS) $(ALE_CLI_OBJS) $(MINIMAL_OBJS) $(RCGA_OBJS) $(ALE_C_OBJS)
DEPS     := $(ALL_OBJS:.o=.d)

# ---- Targets ----
.PHONY: all help clean distclean dirs \
        ale-lib ale-cli minimal rcga ale-c-lib \
        run_minimal_heatmap run_minimal_dataset run_ale run_rcga

all: dirs ale-lib ale-cli $(MINIMAL_BIN) $(RCGA_BIN) $(ALE_C_SO)

help:
	@echo "Targets:"
	@echo "  make all                 -> compila libale.so, ale, minimal_agent (si existe), rcga_train_ale (si existe), libale_c.so (si existe)"
	@echo "  make ale-lib              -> libale.so"
	@echo "  make ale-cli              -> ale"
	@echo "  make minimal              -> minimal_agent"
	@echo "  make rcga                 -> rcga_train_ale"
	@echo "  make ale-c-lib             -> ale_python_interface/libale_c.so (si existe wrapper)"
	@echo ""
	@echo "Variables útiles:"
	@echo "  USE_SDL=1|0               -> compilar con/sin SDL (por defecto 1)"
	@echo "  ROM=...                   -> ruta ROM (por defecto supported/assault.bin)"
	@echo ""
	@echo "Run:"
	@echo "  make run_minimal_heatmap  -> ejecuta minimal_agent ROM heatmap RamFILE.txt"
	@echo "  make run_minimal_dataset  -> ejecuta minimal_agent ROM dataset dataset.txt"
	@echo "  make run_ale              -> ejecuta ale ROM"
	@echo "  make run_rcga             -> ejecuta rcga_train_ale ROM (si existe)"

dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p ale_python_interface

# ---- Build rules (C/C++) ----
$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cxx
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ---- ALE shared library ----
ale-lib: $(ALE_SO)

$(ALE_SO): $(ALE_CORE_OBJS)
	$(CXX) -shared -o $@ $^ $(LDLIBS)

# ---- ALE CLI ----
ale-cli: $(ALE_BIN)

$(ALE_BIN): $(ALE_CLI_OBJS) $(ALE_SO)
	$(CXX) -o $@ $(ALE_CLI_OBJS) -L. -lale $(LDLIBS) -Wl,-rpath,'$$ORIGIN'

# ---- Tu minimal_agent ----
minimal: $(MINIMAL_BIN)

$(MINIMAL_BIN): $(MINIMAL_OBJS) $(ALE_SO)
	$(CXX) -o $@ $(MINIMAL_OBJS) -L. -lale $(LDLIBS) -Wl,-rpath,'$$ORIGIN'

# ---- RCGA trainer (si existe) ----
rcga: $(RCGA_BIN)

$(RCGA_BIN): $(RCGA_OBJS) $(ALE_SO)
	$(CXX) -o $@ $(RCGA_OBJS) -L. -lale $(LDLIBS) -Wl,-rpath,'$$ORIGIN'

# ---- Python C shared lib (si existe wrapper) ----
ale-c-lib: $(ALE_C_SO)

$(ALE_C_SO): $(ALE_C_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) -shared -o $@ $(ALE_C_OBJS) $(LDLIBS)

# ---- Run helpers ----
run_ale: ale-cli
	./$(ALE_BIN) $(ROM)

run_minimal_heatmap: $(MINIMAL_BIN)
	./$(MINIMAL_BIN) $(ROM) heatmap RamFILE.txt

run_minimal_dataset: $(MINIMAL_BIN)
	./$(MINIMAL_BIN) $(ROM) dataset dataset.txt

run_rcga: $(RCGA_BIN)
	./$(RCGA_BIN) $(ROM)

# ---- Clean ----
clean:
	$(RM) $(ALE_SO) $(ALE_BIN) $(MINIMAL_BIN) $(RCGA_BIN)
	$(RM) -r $(BUILD_DIR)

distclean: clean
	$(RM) $(ALE_C_SO)

# ---- Auto include deps ----
-include $(DEPS)
