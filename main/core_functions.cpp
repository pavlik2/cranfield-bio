#include "core_functions.h"
#include <stdlib.h>
void save_SVM(unsigned int length, svm_model** models) {
	system("mkdir svm_model");

	for (int i = 0; i < length; i++) {
		 char model_dir[255];
		sprintf(model_dir,"svm_model/model%d",i);

		svm_save_model(model_dir, models[i]);
	}
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

svm_model** load_SVM(size_t& SVM_COUNT) {
	svm_model** ret = Malloc(svm_model*,1000);
	DIR* dir;
	dirent* pdir;
 char model_dir[255] = "svm_model/";
	dir = opendir(model_dir);     // open current directory
	unsigned int count = 0;
	while (pdir = readdir(dir)) {
		char model_dir[255] = "svm_model/";
	const	char* file = pdir->d_name;
		svm_model* model = NULL;
		if (strcmp(file, ".") != 0 && strcmp(file, "..") != 0){
			char* model_path = strcat(model_dir,file);
			model = svm_load_model(model_path);
		}
		if (model != NULL) {
			ret[count++] = model;
		}
	}
	closedir(dir);
	SVM_COUNT=count;
	return ret;
}

void divide_sets(float* array1, double* array2, int size, int col1,
		int test_size, float*& trainData, double*& trainClass, float*& testData,
		double*& testClass) {

	int* numbers = random_number(size, test_size);

	const size_t trainDataSize = size - test_size;

	testData = new float[test_size * col1 + 1];
	testClass = new double[test_size + 1];

	trainClass = new double[trainDataSize + 1];

	trainData = new float[trainDataSize * (col1 + 1)];

	int trainPos = 0;
	int testPos = 0;
	for (int i = 0; i < size; i++) {

		if (!isInArray(i, numbers, test_size)) {
			for (int c = 0; c < col1; ++c) {
				trainData[trainPos * col1 + c] = array1[i * col1 + c]; //---CHECK HERE-debug here
			}

			trainClass[trainPos++] = (double) array2[i];		//--debug here
		} else {
			int number = numbers[testPos];
			for (int c = 0; c < col1; c++) {
				testData[testPos * col1 + c] = array1[number * col1 + c];//--debug here
			}
			testClass[testPos++] = array2[number];
		}
	}

}

void divide_sets(float* array1, float* array2, int size, int col1,
		int test_size, float*& trainData, double*& trainClass, float*& testData,
		double*& testClass) {

	int* numbers = random_number(size, test_size);

	const size_t trainDataSize = size - test_size;

	testData = new float[test_size * col1 + 1];
	testClass = new double[test_size + 1];

	trainClass = new double[trainDataSize + 1];

	trainData = new float[trainDataSize * (col1 + 1)];

	int trainPos = 0;
	int testPos = 0;
	for (int i = 0; i < size; i++) {

		if (!isInArray(i, numbers, test_size)) {
			for (int c = 0; c < col1; ++c) {
				trainData[trainPos * col1 + c] = array1[i * col1 + c];//---CHECK HERE-debug here
			}

			trainClass[trainPos++] = (double) array2[i];		//--debug here
		} else {
			int number = numbers[testPos];
			for (int c = 0; c < col1; c++) {
				testData[testPos * col1 + c] = array1[number * col1 + c];//--debug here
			}
			testClass[testPos++] = array2[number];
		}
	}

}
