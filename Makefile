CXX = g++
CXXFLAGS = -O3 -Wall -std=c++11 -march=native -pthread
LIBS = -larmadillo -lm

TARGET = fractalwalk
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LFLAGS)

.PHONY: clean

clean:
	rm *.o $(TARGET)
