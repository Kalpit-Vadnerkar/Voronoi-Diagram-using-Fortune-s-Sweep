NVCC=nvcc 

OPENCV_INCLUDE_PATH="$(OPENCV_ROOT)/include/opencv4"

OPENCV_LD_FLAGS = -L $(OPENCV_ROOT)/lib64 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

CUDA_INCLUDEPATH=/usr/local/cuda/include

NVCC_OPTS=-arch=sm_30 
GCC_OPTS=-std=c++11 -g -O3 -Wall 
CUDA_LD_FLAGS=-L /usr/local/cuda/lib64 -lcuda -lcudart

final: main.o brute_kernel.o
	g++ -o brute main.o brute_kernel.o $(CUDA_LD_FLAGS) $(OPENCV_LD_FLAGS)

main.o:main.cpp brute_kernel.h utils.h 
	g++ -c $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDE_PATH) main.cpp 

brute_kernel.o: brute_kernel.cu brute_kernel.h utils.h
	$(NVCC) -c brute_kernel.cu $(NVCC_OPTS)

clean:
	rm *.o brute
