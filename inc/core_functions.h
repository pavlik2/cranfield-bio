#include "add.h"
#include "svm_function.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <string>
svm_model** load_SVM(size_t& SVM_COUNT);

void divide_sets(float* array1, float* array2, int size, int col1,
		int test_size, float*& trainData, double*& trainClass, float*& testData,
		double*& testClass);

void divide_sets(float* array1, double* array2, int size, int col1,
		int test_size, float*& trainData, double*& trainClass, float*& testData,
		double*& testClass) ;

void save_SVM(unsigned int length,svm_model** models);
