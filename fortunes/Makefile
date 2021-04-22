NVCC=nvcc 

OPENCV_INCLUDE_PATH="$(OPENCV_ROOT)/include/opencv4"

OPENCV_LD_FLAGS = -L $(OPENCV_ROOT)/lib64 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

CUDA_INCLUDEPATH=/usr/local/cuda/include

NVCC_OPTS=-arch=sm_30 
GCC_OPTS=-std=c++11 -g -O3 -Wall 
CUDA_LD_FLAGS=-L /usr/local/cuda/lib64 -lcuda -lcudart

final: main.o parabola.o voronoi.o
	g++ -o fortune main.o VParabola.o Voronoi.o

main.o: main.cpp Voronoi.h VPoint.h 
	g++ -c $(GCC_OPTS) main.cpp

parabola.o: VParabola.cpp VParabola.h VPoint.h
	g++ -c $(GCC_OPTS) VParabola.cpp

voronoi.o: Voronoi.cpp Voronoi.h
	g++ -c $(GCC_OPTS) Voronoi.cpp

clean:
	rm *.o fortune
