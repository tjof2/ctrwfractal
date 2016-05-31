CXX = g++
CXXFLAGS = -O3 -Wall -std=c++11 -march=native
LIBS = -larmadillo -lm

TARGET = percolation
SRCS = perc.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LFLAGS)

.PHONY: clean

clean:
	rm *~ *.o  $(TARGET)
