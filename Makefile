CXXFLAGS =	$(OPT)  -static -Wall -fPIC -fmessage-length=0 -fopenmp  -I inc/ -I ann/include/ -I libcudaann/ -I /usr/local/cuda/include/ -I common/inc -I /usr/include/ -I /usr/lib/gcc/x86_64-linux-gnu/4.6/include#-fno-stack-protector
#O3-g3-g -std=gnu++0x
#-gencode arch=compute_30,code=sm_30
OPT = -O3 -DOMP -fopenmp
CUDA_OPTS = -c -gencode arch=compute_30,code=sm_30 -I inc/ -I /usr/local/cuda/include/ -I"/usr/local/cuda/samples/common/inc" -I common/inc -m64
OBJS = CPP_BIO.o core_functions.cpp.o  main/add.cpp.o main/core.cpp.o ann/doublefann.c.o
CUDAANN_OBJS = libcudaann/FeedForwardNN.cpp.o libcudaann/FeedForwardNNTrainer.cpp.o libcudaann/ActivationFunctions.cpp.o libcudaann/ErrorFunctions.cpp.o libcudaann/GAFeedForwardNN.cpp.o libcudaann/FloatChromosome.cpp.o libcudaann/CudaActivationFunctions.cu.o libcudaann/CudaErrorFunctions.cu.o libcudaann/LearningSet.cpp.o
CXX = g++
LIBS = -L/usr/local/cuda/lib64/ -L/lib -L/usr/lib -L.  -lboost_program_options 
CULIBS = -Llib -lcutil_x86_64 -lcublas -lcuda -lcudart 
CU_COMP = /usr/local/cuda/bin/nvcc
TARGET =	CPP_BIO

#svm_function.cpp.o $(ADDOBJS) libsvm-dense-3.17/svm.cpp.o
data-converter:transpose
	
library:  
	#R CMD SHLIB library.cpp libsvm-dense-3.17/svm.cpp
	g++ -fopenmp -I/usr/share/R/include -fpic  -O3 -shared  -g library.cpp libsvm-dense-3.17/svm.cpp -o library.so -std=gnu++0x -lR -pipe -DOMP
#library: g++ -fopenmp -I/usr/share/R/include -fpic  -O3 -shared  -g library.cpp libsvm-dense-3.17/svm.cpp -o library.so -std=gnu++0x -lR -pipe -DOMP


transpose:
	g++ -O0 -g3 data-converter.cpp main/add.cpp -o data-converter

gpufloat: CXXFLAGS+= -DFLOAT1g 

gpufloat: CUDA_OPTS+= -DFLOAT1g 

gpufloat: gpu

all2float: CXXFLAGS+= -DFLOAT1g 

all2float: CUDA_OPTS+= -DFLOAT1g 

all2float: all2

gpu: OBJS+= main/gpu.cu.o

gpu: main/gpu.cu.o
gpu: OBJS+= $(CUDAANN_OBJS)
gpu: CXXFLAGS+= -DGPU_TEST

gpu:	OPT = -O3

gpu:	CUDA_OPTS+=-O3 -gencode arch=compute_30,code=sm_30 -DGPU_TEST -lineinfo  -maxrregcount=63 -Xptxas -v --ptxas-options=-v


gpu: LIBS+= $(CULIBS)
gpu: 	OBJS+= main/gpu_core.cpp.o

gpu:	main/gpu_core.cpp.o

gpu:	$(TARGET)

gpudebug: OBJS+= main/gpu.cu.o

gpudebug: main/gpu.cu.o

gpudebug:: CXXFLAGS+= -DGPU_TEST

gpudebug:	OPT = -O0

gpudebug:	CUDA_OPTS+=-O0 -g -G -pg -DDEBUG -gencode arch=compute_30,code=sm_30 -DGPU_TEST -lineinfo  -maxrregcount=63 -Xptxas -v --ptxas-options=-v

gpudebug:	CXXFLAGS+= -g3 -pg -DDEBUG

gpudebug: 	OBJS+= main/gpu_core.cpp.o

gpudebug:	main/gpu_core.cpp.o

gpudebug:	$(TARGET)


debugcpp: CXXFLAGS+= -g3 -pg -DDEBUG -DFLOAT1g

debugcpp:	OPT = -O0 -pg

debugcpp:	CUDA_OPTS+=-O0 -G -g -DDEBUG -DFLOAT1g -maxrregcount=63

debugcpp: OBJS+= main/gpu2.cu.o

debugcpp: main/gpu2.cu.o

debugcpp: OBJS+= svm_function3.cpp.o

debugcpp: svm_function3.cpp.o


debugcpp:	$(TARGET)


all:	OBJS+= libsvm-dense-3.17/svm.cpp.o

all:	OPT+= -DLIBSVM

all: 	libsvm-dense-3.17/svm.cpp.o

all:	$(TARGET)

all2:	OPT = -O3 

all2:	CUDA_OPTS+=-O3 -maxrregcount=63

all2: OBJS+= main/gpu2.cu.o

all2: main/gpu2.cu.o

all2:	OBJS+= svm_function3.cpp.o

all2:	LIBS+= $(CULIBS)

all2: 	svm_function3.cpp.o

all2:	$(TARGET)

debug:	OPT = -O0 -pg

debug:	CUDA_OPTS+=-O0 -G -g -DDEBUG -DLIBSVM

debug:	CXXFLAGS+= -g3 -pg -DDEBUG -DLIBSVM

debug:	OBJS+= libsvm-dense-3.17/svm.cpp.o

debug:	libsvm-dense-3.17/svm.cpp.o

debug:	$(TARGET)

transform: OPT+= -DTRANSFORM
transform: OBJS+=svm_function.cpp.o
transform: svm_function.cpp.o
#include "../svm_function.cpp"
transform: $(TARGET)

$(TARGET):	$(OBJS)
	$(CXX) $(OPT) -o $(TARGET) $(OBJS)  $(LIBS) 

.SUFFIXES: .o .cc .cxx .cpp .cu

.cu.o:
	$(CU_COMP) $(OPT) -o "$@" "$<"
libsvm-dense-3.17/svm.cpp.o:libsvm-dense-3.17/svm.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) -o "$@" "$<"
gpu_fake.cpp.o:gpu/gpu_fake.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) -o "$@" "$<"
main/add.cpp.o:main/add.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) -o "$@" "$<"
main/gpu_core.cpp.o:main/gpu_core.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) -o "$@" "$<"
main/core.cpp.o:main/core.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) -o "$@" "$<"
core_functions.cpp.o:main/core_functions.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) -o "$@" "$<"
ann/doublefann.c.o:ann/doublefann.c
	$(CXX) $(OPT) -c $(CXXFLAGS) -o "$@" "$<"
svm_function.cpp.o:svm_function.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) -o "$@" "$<"
svm_function.cpp2.o:svm_function2.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) -o "$@" "$<"
svm_function3.cpp.o:svm_function3.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) -o "$@" "$<"
main/gpu.cu.o:main/gpu.cu
	$(CU_COMP)  $(CUDA_OPTS) -o "$@" "$<"
main/gpu2.cu.o:main/gpu2.cu
	$(CU_COMP)  $(CUDA_OPTS) -o "$@" "$<"
main/gpu3.cu.o:main/gpu3.cu
	$(CU_COMP)  $(CUDA_OPTS) -o "$@" "$<"
main/gpu4.cu.o:main/gpu4.cu
	$(CU_COMP)  $(CUDA_OPTS) -o "$@" "$<"
libcudaann/FeedForwardNN.cpp.o:libcudaann/FeedForwardNN.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) $< -o $@
libcudaann/LearningSet.cpp.o:libcudaann/LearningSet.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) $< -o $@  
libcudaann/FeedForwardNNTrainer.cpp.o:libcudaann/FeedForwardNNTrainer.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) $< -o $@
libcudaann/ActivationFunctions.cpp.o:libcudaann/ActivationFunctions.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) $< -o $@
libcudaann/ErrorFunctions.cpp.o:libcudaann/ErrorFunctions.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) $< -o $@
libcudaann/GAFeedForwardNN.cpp.o:libcudaann/GAFeedForwardNN.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) $< -o $@
libcudaann/FloatChromosome.cpp.o:libcudaann/FloatChromosome.cpp
	$(CXX) $(OPT) -c $(CXXFLAGS) $< -o $@
libcudaann/CudaActivationFunctions.cu.o:libcudaann/CudaActivationFunctions.cu
	$(CU_COMP) -c  $(CUDA_OPTS) -o "$@" "$<"
libcudaann/CudaErrorFunctions.cu.o:libcudaann/CudaErrorFunctions.cu
	$(CU_COMP) -c  $(CUDA_OPTS) -o "$@" "$<"
clean: cleanq
	rm -f $(OBJS) $(TARGET) svm_function3.cpp.o main/gpu_core.cpp.o main/gpu3.cu.o main/gpu2.cu.o svm_function.cpp2.o main/gpu.cu.o svm_function.cpp.o libsvm-dense-3.17/svm.cpp.o
	
cleanq:
	rm -f $(TARGET) main/gp*.o
