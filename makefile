DEPS=methods/Pregel.cpp methods/CFDGraph.cpp main.cpp src/GraphConstruction.cpp utils/Utils.cpp

OSFLAG = $(shell uname -s)
ifeq ($(OSFLAG),Darwin)
	CC =clang++
	CFLAGS =--std=c++11 -Xpreprocessor -fopenmp -lomp -g 
else ifeq ($(OSFLAG),Linux)
	CC=/usr/bin/g++
	CFLAGS=-std=c++11  -fopenmp -g 
endif
CFDGraph: $(DEPS)
	$(CC) $(CFLAGS) $(DEPS) -o CFDGraph
#	$(CC) $(CFLAGS) $(DEPS) -o CFDGraph
clean:
	rm -f CFDGraph