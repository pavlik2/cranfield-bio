#include<iostream>
#include <string>
#include <cmath>
#include "add.h"
#include "gpu.h"

void gpu_process(float Cstart, float gStart, float C, float Cstep, float gamma,
		float gammaStep, int svm_count, int iterations, size_t test_size,
		int col1, int row1, float* array1, float* array2,
		bool classiffication) {
	float a = 0, b = 0;
	array1 = scale2(row1, col1, array1, 1, -1, a, b);
	srand(time(NULL));

	for (size_t svm_machine = 0; svm_machine < svm_count; svm_machine++) {

		int* numbers = random_number(row1, test_size, (iterations + 1));
		const size_t trainDataSize = row1 - test_size;
		//int trainDataSize=row1-test_size;
		float* testData = new float[test_size * col1 + 1];
		float* testClass = new float[test_size + 1];

		
#ifdef FLOAT1g
float
#else
double
#endif

* trainClass = new 
#ifdef FLOAT1g
float
#else
double
#endif

[trainDataSize + 1];

		float* trainData = new float[trainDataSize * (col1 + 1)];

		for (size_t i = 0; i < test_size; i++) {
			int number = numbers[i];
			for (int c = 0; c < col1; c++) {
				testData[i * col1 + c] = array1[number * col1 + c];	//--debug here
			}
			testClass[i] = array2[number];		//--debug here

		}

		int trainPos = 0;

		for (int i = 0; i < row1; i++) {

			if (!isInArray(i, numbers, test_size)) {
				for (int c = 0; c < col1; ++c) {
					trainData[trainPos * col1 + c] = array1[i * col1 + c];//---CHECK HERE-debug here
				}

				trainClass[trainPos++] = array2[i];		//--debug here
			}
		}

		int tot_g = gamma / gammaStep;

		int tot_C = C / Cstep+Cstep;
		int length_total = tot_g * tot_C;

		svm_p* array_param = new svm_p[length_total*2];

		int idx = 0;
		for (float i = Cstart; i <= C; i += Cstep)
			for (float j = gStart; j <= gamma; j += gammaStep) { //if (idx<length_total) {
				array_param[idx].C = i;
				array_param[idx].gamma = j;
				idx++;
//	}
			}
		length_total=idx;
		
#ifdef FLOAT1g
float
#else
double
#endif
* dec_values =

		RunGPU(trainDataSize, trainClass, trainData, col1, testData,
				(int) test_size, array_param, length_total,classiffication);




		for (int i = 0; i < length_total; i++) {
			int predicted_total = 0;
			for (unsigned int j = 0; j < test_size; j++) {
#ifdef DEBUG
				printf("Original:%f Predicted: %f\n", testClass[j],
						dec_values[j]);
#endif
				//printf("%d ",(int)dec_values[j*i+i]);
				if (classiffication) {
				if (testClass[j] == dec_values[test_size*i+j])
					predicted_total++; }
				else {if (abs((testClass[j]) - abs(dec_values[test_size*i+j]))<=1.1)
					predicted_total++;}

			}

			float performance = ((float) predicted_total / (float)test_size) * 100;

			printf("C - %f gamma - %.2f Performance: %f\n", array_param[i].C,
					array_param[i].gamma, performance);

		}

//		RunGPU(trainData,trainDataSize,col1,trainClass,testData,test_size);
		exit(0);

	}

}
