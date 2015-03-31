/*
 * core.h
 *
 *  Created on: 1 Jul 2013
 *      Author: pavel
 */

#ifndef CORE_H_
#define CORE_H_
#include "svm_function.h"
struct svRet {
	float* meanPerformanceRaw;
	float* meanPerformanceAdaBoost;
	int count;
};

svm_model**  do_training(float Cstart, float gStart, float C, float Cstep,
		float gamma, float gammaStep, size_t SVM_COUNT, int iterations1,
		 int col1, int row1, float* array1, double* array2,
		bool classiffication, bool gpu);

svm_model**  do_training_fast(float C,float Cstep,float gamma,float gammaStep,
		size_t SVM_COUNT,int iterations1,size_t test_size
		,int col1,int row1,float* array1,float* array2);

svRet analysis(size_t SVM_COUNT, size_t test_size, int col1,
		 svm_model** models,float* testData,double* testClass,bool classiffication);

fann** do_training_ann(
		size_t SVM_COUNT, size_t test_size, int col1, int row1,
		float* array1, float* array2,bool classification,bool cascade
		,unsigned int num_layers,unsigned int num_neurons_hidden,
		float desired_error,unsigned int max_epochs ,
				unsigned int max_neurons
		);
svRet analysis_neural(int SVM_COUNT, size_t test_size, int col1, int row1,
		float* array1, float* array2, fann** anns);

#ifdef GPU_TEST
FeedForwardNN** do_training_ann_gpu(size_t SVM_COUNT, size_t test_size,
		int col1, int row1, float* array1, float* array2, bool classification,
		bool cascade, unsigned int num_layers, unsigned int num_neurons_hidden,
		float desired_error, unsigned int max_epochs,
		unsigned int max_neurons);
#endif

#endif /* CORE_H_ */
