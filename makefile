# define compiler and path of libs
# macOS (Darwin) needs explicit paths for Homebrew's libomp/eigen/boost, since
# none of them sit on the compiler's default search path the way apt's do on
# Linux. MSYS2/MinGW-w64 on Windows reports a MINGW64_NT-* uname; its pacman
# packages land on the default search path like apt's do, but its boost
# package names the library with a "-mt" (multi-threaded) suffix instead of
# the plain name Ubuntu/Homebrew use.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OMPFLAGS   = -Xpreprocessor -fopenmp -I$(shell brew --prefix libomp)/include
    OMPLIBS    = -L$(shell brew --prefix libomp)/lib -lomp
    EIGENFLAGS = -I$(shell brew --prefix eigen)/include
    BOOSTFLAGS = -I$(shell brew --prefix boost)/include
    BOOSTLIBS  = -L$(shell brew --prefix boost)/lib -lboost_program_options
else ifneq (,$(findstring MINGW,$(UNAME_S)))
    OMPFLAGS   = -fopenmp
    OMPLIBS    = -fopenmp
    EIGENFLAGS =
    BOOSTFLAGS =
    BOOSTLIBS  = -lboost_program_options-mt
else
    OMPFLAGS   = -fopenmp
    OMPLIBS    = -fopenmp
    EIGENFLAGS =
    BOOSTFLAGS =
    BOOSTLIBS  = -lboost_program_options
endif
# c++14 (not c++11): current Homebrew Eigen (5.x) requires c++14 to compile;
# c++14 is a strict superset of c++11 so this doesn't affect Linux/Windows.
CPPFLAGS = --std=c++14 -w -I $(DIR_INC) $(OMPFLAGS) $(EIGENFLAGS) $(BOOSTFLAGS)
LDLIBS = $(BOOSTLIBS) $(OMPLIBS)
# define variable path
DIR_INC = include
DIR_SRC = HH HHcon Lorenz Lcon Rcon Logistic
DIR_BIN = bin

vpath %.cpp $(DIR_SRC)
vpath %.h   $(DIR_INC)
vpath %.h   $(DIR_SRC)

HEADERS_COMMON := $(notdir $(wildcard $(DIR_INC)/*.h))
HEADERS_HH := $(notdir $(wildcard HH/*.h))
HEADERS_Causality := $(notdir $(wildcard Causality/*.h))
SRCS_Causality = $(wildcard Causality/*.cpp)
OBJS_Causality = $(SRCS_Causality:.cpp=.o)
HEADERS_HHcon := $(notdir $(wildcard HHcon/*.h))
HEADERS_Lorenz := $(notdir $(wildcard Lorenz/*.h))
HEADERS_Lcon := $(notdir $(wildcard Lcon/*.h))
HEADERS_Rossler := $(notdir $(wildcard Rossler/*.h))
HEADERS_Rcon := $(notdir $(wildcard Rcon/*.h))
BIN := $(DIR_BIN)/simHH $(DIR_BIN)/simHHcon $(DIR_BIN)/simLorenz $(DIR_BIN)/simLcon $(DIR_BIN)/simRcon $(DIR_BIN)/simLogistic

.PHONY : all
all : $(BIN)

$(DIR_BIN)/calCausality : $(OBJS_Causality) $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_Causality)
	$(CXX) $(CPPFLAGS) -o $(DIR_BIN)/calCausality $(OBJS_Causality) $(LDLIBS)

$(DIR_BIN)/simHH : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_HH) 
	$(CXX) $(CPPFLAGS) -O2 HH/main.cpp -o $(DIR_BIN)/simHH $(LDLIBS)

$(DIR_BIN)/simHHcon : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_HHcon) 
	$(CXX) $(CPPFLAGS) HHcon/main.cpp -o $(DIR_BIN)/simHHcon $(LDLIBS)

$(DIR_BIN)/simLorenz : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_Lorenz) 
	$(CXX) $(CPPFLAGS) -O2 Lorenz/main.cpp -o $(DIR_BIN)/simLorenz $(LDLIBS)

$(DIR_BIN)/simLcon : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_Lcon) 
	$(CXX) $(CPPFLAGS) Lcon/main.cpp -o $(DIR_BIN)/simLcon $(LDLIBS)

$(DIR_BIN)/simRossler : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_Rossler) 
	$(CXX) $(CPPFLAGS) -O2 Rossler/main.cpp -o $(DIR_BIN)/simRossler $(LDLIBS)

$(DIR_BIN)/simRcon : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_Rcon) 
	$(CXX) $(CPPFLAGS) -O2 Rcon/main.cpp -o $(DIR_BIN)/simRcon $(LDLIBS)

$(DIR_BIN)/simFN : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_FN) 
	$(CXX) $(CPPFLAGS) FN/main.cpp -o $(DIR_BIN)/simFN $(LDLIBS)

$(DIR_BIN)/simML : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_ML) 
	$(CXX) $(CPPFLAGS) ML/main.cpp -o $(DIR_BIN)/simML $(LDLIBS)

$(DIR_BIN)/simLogistic : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_Logistic) 
	$(CXX) $(CPPFLAGS) Logistic/main.cpp -o $(DIR_BIN)/simLogistic $(LDLIBS)

.PHONY : debug
debug : CPPFLAGS += -DDEBUG
debug : $(BIN)

$(DIR_BIN) : 
	@mkdir -p $@

.PHONY : clean
clean:
	rm -rf $(DIR_BIN)
