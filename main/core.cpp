/*
 * core.cpp
 *
 *  Created on: 1 Jul 2013
 *      Author: pavel
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "add.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <utility>
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <omp.h>
#include <string.h>
#include "doublefann.h"
#include "core_functions.h"


#include "svm_function.h"

#include "gpu.h"
#define ADA_BOOST
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
//#define OMP

//#define TEST_SPEED
#ifdef DEBUG
#define DEBUG_SVM

#define PRINT_ALL
#endif
using namespace std;

struct svRet {
	float* meanPerformanceRaw;
	float* meanPerformanceAdaBoost;
	int count;
};


void string_stdout(const char *s) {
}
;
bool lim = true;

float predict_error_th = 1.1;
svm_model** do_training(float Cstart, float gStart, float C, float Cstep,
		float gamma, float gammaStep, size_t SVM_COUNT, int iterations1,
		 int col1, int row1, float* array1, double* array2,
		bool classiffication, bool gpu) {
#ifndef GPU_TEST
#ifndef LIBSVM
	if (gpu)
	initGPU(row1 * 2, col1);
#endif
#endif
	//bool classiffication=false;

	int max_iter = iterations1;
	size_t test_size = row1/4;
#ifdef DEBUG
	printf("Rows train %d Test: %d",row1,test_size);
#endif

	svm_model** models;
	svm_model *model;


	printf("Test size 1:%d Train size:%d\n",test_size,row1);


	models = Malloc(svm_model*,SVM_COUNT);
#ifndef GPU_TEST
#ifndef TRANSFORM
	svm_set_print_string_function(&string_stdout);

#endif
#endif
#ifndef TEST_SPEED
#ifdef OMP
	int iCPU = omp_get_num_procs();
	omp_set_num_threads(iCPU);
#pragma omp parallel for private (model)
#endif
	//TODO PARAMETERS
	for (size_t svm_machine = 0; svm_machine < SVM_COUNT; svm_machine++) {
		float modelPerformance = 0.0;
		bool modelFree = false;
		for (float currentC = Cstart; currentC < C; currentC += Cstep) {

			for (float currentGTamma = gStart; currentGTamma < gamma;
					currentGTamma += gammaStep) {
#else
				float currentC = 1;
				float currentGTamma = 0.1;
#endif
				float currentPerformance = 0.0;

//Start-----------------------------------

				float predicted_total = 0;
				svm_problem prob;		// set by read_problem
				svm_node* svm_n = NULL;

				for (int iterations = 0; iterations < iterations1;
						iterations++) {
					srand(time(NULL));
					svm_parameter param;		// set by parse_command_line

					const size_t trainDataSize = row1 - test_size;

					float* testData=NULL;

					double* testClass=NULL;

					double* trainClass=NULL;

					float* trainData=NULL;

					divide_sets(array1, array2, row1, col1, test_size,
							trainData, trainClass, testData, testClass);

#ifdef DEBUG_SVM
					cout << "Train Class:" << endl;
					for (size_t i = 0; i < trainDataSize; i++)
					if (trainClass[i] < 1) {
						printf("%d %f \n", i, trainClass[i]);
					}
					printf("Data setup %d complete\n", iterations);
#endif
					if (classiffication)
						param.svm_type = C_SVC;
					else
						param.svm_type = EPSILON_SVR; //C_SVC;

					param.kernel_type = RBF;
					param.degree = 3;
					param.gamma = currentGTamma;	// 1/num_features
					param.coef0 = 0;
					param.nu = 0.5;
					param.cache_size = 1;
					param.C = currentC;
					param.eps = 1e-3;
					param.p = 0.1;
					param.shrinking = 1;
					param.probability = 0;

					param.nr_weight = 0;
					param.weight_label = NULL;
					param.weight = NULL;
					//cross_validation = 0;
//---------------------------------------------------------
					prob.l = trainDataSize;
					prob.y = trainClass;

					svm_n = new svm_node[trainDataSize]; //memory leak fixed?
#ifdef DEBUG
							cout << "Param Ident" <<endl;
#endif

					for (size_t i = 0; i < trainDataSize; i++) {

						//svm_n[i].values = new double[col1 + 1];	//-debug
						svm_n[i].dim = col1;
						//	for (int c = 0; c < col1; ++c) {

						svm_n[i].values = scale(i, col1, trainData, 1.0, -1.0);

						//	(double) trainDa#define TRANSFORMta[i * col1 + c];//---CHECK HERE-debug here

						//	}

					}

#ifdef DEBUG
					cout << "Param Ident2" <<endl;
#endif
					prob.x = svm_n;

					const char *error_msg;
#ifndef GPU_TEST
					error_msg = svm_check_parameter(&prob, &param);
					if (error_msg) {
						fprintf(stderr, "Error: %s\n", error_msg);
						exit(1);
					}
#endif
#ifdef DEBUG
					cout << "Param Ident3" <<endl;
#endif
#ifndef GPU_TEST
#ifndef TRANSFORM2
					model = svm_train(&prob, &param, gpu);
#else
					model = svm_train2(&prob, &param);
#endif

#endif

#ifdef VALIDATION
					double *target = Malloc(double,prob.l);
					//----------------------------
					int total_correct = 0;
					double total_error = 0;
					double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
					svm_cross_validation(&prob,&param,15,target);
					if(param.svm_type == EPSILON_SVR ||
							param.svm_type == NU_SVR)
					{
						for(int i=0;i<prob.l;i++)
						{
							double y = prob.y[i];
							double v = target[i];
							total_error += (v-y)*(v-y);
							sumv += v;
							sumy += y;
							sumvv += v*v;
							sumyy += y*y;
							sumvy += v*y;
						}

						printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
						printf("Cross Validation Squared correlation coefficient = %g\n",
								((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
								((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
						);
					}

					//----------------------------
					delete[] target;
#endif

					double

					* dec_values = Malloc(
							double

							,test_size + 1);

					//--	Test Data
					//		svm_node* svm_t = new svm_node[test_size + 1];
					svm_node* svm_t = new svm_node;

					for (size_t i = 0; i < test_size; i++) {

						//	svm_t->values = new double[col1 + 1];	//-debug
						svm_t->dim = col1;

						//	for (int c = 0; c < col1; ++c) {

						svm_t->values =							//[c] =
								scale(i, col1, testData, 1.0, -1.0);

						//		(double) testData[i * col1 + c];//---CHECK HERE-debug here
						//	}
#ifndef GPU_TEST
						dec_values[i + 1] = svm_predict(model, svm_t);
#endif
						delete[] svm_t->values;
					}

					delete[] svm_t; //[]//TODO

					//	cout << "Test" << endl;
					//	for (int i = 0; i < test_size; i++)
					//		dec_values[i] = svm_predict(model, &svm_t[i]);

					//svm_predict(model, svm_t);

					//	svm_predict_values(model, svm_t, dec_values);

#ifdef DEBUG_SVM
					for (size_t i = 1; i<= test_size; i++)
					printf("Actual: %f Predicted: %f\n", testClass[i-1], dec_values[i]);
#endif
					predicted_total = 0;

					if (classiffication)
						predict_error_th = 0.5;

					for (size_t i = 1; i <= test_size; i++)
						if (abs(testClass[i - 1] - abs(dec_values[i]))
								<= predict_error_th)
							predicted_total += 1.0;
						else if (classiffication)
							if (dec_values[i] < 1.0)
								predicted_total += 1.0;
					free(dec_values);
					//delete[] dec_values;
					//delete dec_values;
#ifdef DEBUG
					cout << "Param Ident4" <<endl;
#endif
#ifndef GPU_TEST
					if (iterations != (max_iter - 1)) {
						svm_free_and_destroy_model(&model);
						for (int i = 0; i < prob.l; i++) //memory leak (fixed?)
							delete[] svm_n[i].values;
						delete[] prob.x;
					}

					svm_destroy_param(&param);
#endif
#define TRANSFORM
#ifdef DEBUG
					cout << "Param Ident5" <<endl;
#endif
					//delete prob;
					delete[] testData;
					delete[] testClass;

					delete[] trainClass;
					delete[] trainData;

#ifdef TEST_SPEED
					printf("Check svm: currentC %f currentGTamma %f iteration %d \n",
							currentC, currentGTamma, iterations);
#endif

					float performance = (predicted_total / test_size) * 100;
					currentPerformance += performance;

				}
				currentPerformance = currentPerformance / max_iter;
#ifndef TEST_SPEED
#ifdef PRINT_ALL
				printf(
						"Check svm: currentC %f currentGTamma %f prediction: %.2f% \n",
						currentC, currentGTamma, currentPerformance);
#endif
#ifdef OMP
#pragma omp critical
#endif
#ifndef GPU_TEST
				if (currentPerformance > modelPerformance) {
					modelPerformance = currentPerformance;
					if (modelFree) {

						svm_free_and_destroy_model(&models[svm_machine]);

					}

					models[svm_machine] = model;
					modelFree = true;
				} else {
					//delete[] model->SV;
					svm_free_and_destroy_model(&model);
					for (int i = 0; i < prob.l; i++) //memory leak (fixed?)
						delete[] svm_n[i].values;
				}
#endif
			}

		}
		printf("Best Performance in svm %u= %f\%\n", svm_machine,
				modelPerformance);
	}
#endif
#ifndef GPU_TEST
#ifndef LIBSVM
	if (gpu)
	freeGPU();
#endif
#endif
	return models;
};

#ifndef GPU_TEST

#endif
svRet analysis(size_t SVM_COUNT, size_t test_size, int col1,
		 svm_model** models,float* testData,double* testClass,bool classiffication) {
	svRet returns;
	returns.count = SVM_COUNT;
	returns.meanPerformanceAdaBoost = new float[SVM_COUNT];
	returns.meanPerformanceRaw = new float[SVM_COUNT];

	svm_model *model;
	if (classiffication)
							predict_error_th = 0.5;


	//TODO ADA_START
//---------------------Plot means
	double** prediction_table = Malloc(double*,SVM_COUNT);



	float meanPerformanceRaw[SVM_COUNT];
	//float meanPerformanceRaw[SVM_COUNT];
	float meanPerformanceAdaBoost[SVM_COUNT];

//AdaBoost
	float Cerror[SVM_COUNT];
	float SampleErrors[test_size];
	double SvmAdaBoost[SVM_COUNT];
	float totalPerformanceS[SVM_COUNT];
	for (size_t i = 0; i < test_size; i++)
		SampleErrors[i] = 1.0 / (float) test_size;

	for (size_t i = 0; i < SVM_COUNT; i++) {
		//float totalPerformance[SVM_COUNT];
		double* dec_values;		//= Malloc(double,test_size + 1);
		model = models[i];
		//double* maxVote = new double[current+1];
		dec_values = Malloc(double,test_size + 1);
		svm_node* svm_t = new svm_node;
		for (size_t i1 = 0; i1 < test_size; i1++) {

			svm_t->dim = col1;

			svm_t->values =							//[c] =
					scale(i1, col1, testData, 1.0, -1.0);

			//		(double) testData[i * col1 + c];//---CHECK HERE-debug here
#ifndef GPU_TEST
			dec_values[i1 + 1] = svm_predict(model, svm_t);
#endif
			delete[] svm_t->values;

		}
		delete svm_t;
		prediction_table[i] = dec_values;
/////----------------
		//Total performance raw
		totalPerformanceS[i] = performance(prediction_table[i], testClass,
				test_size, predict_error_th);

		meanPerformanceRaw[i] = mean(totalPerformanceS, i + 1);

		/////////----------------
		for (size_t current = 0; current <= i; current++) {

			//dec_values;
			int* indexes = NULL;
			int ind_length = wrong_compare(prediction_table[current], testClass,
					test_size, &indexes);
			float* Error = new float[ind_length];
			for (int val = 0; val < ind_length; val++) {
				Error[val] = SampleErrors[indexes[val]];
			}
			Cerror[i] = sum(Error, ind_length);

		}

		double w = 1.0 / 3.0 * log((1 - Cerror[i]) / Cerror[i]);
		SvmAdaBoost[i] = w;

		for (size_t t = 0; t < test_size; t++) {

			if (!compare(testClass[t], prediction_table[i][t + 1])) {
				SampleErrors[t] *= pow(2.718282, -w);
			}

		}
//TODO _REMOVE

		for (size_t t = 0; t < test_size; t++) {

			SampleErrors[t] = SampleErrors[t] / (sum(SampleErrors, test_size));

		}

	}

	for (size_t i = 0; i < SVM_COUNT; i++) {
		float totalPerformance[SVM_COUNT];

		double* maxVote = new double[test_size];

		for (size_t pred_n = 0; pred_n < test_size; pred_n++) {
			float winnerWeight = 0;
			for (size_t current = 0; current <= i; current++) {

				int compare = 0;

				float CurrentVoteWeigth = SvmAdaBoost[current];
				double currentVote = prediction_table[current][pred_n + 1];
				//float currentVoteWeight=0;

				for (size_t n = 0; n <= current; n++) {
					double pre_vote = (prediction_table[n][pred_n + 1]);
					if (abs(pre_vote - currentVote) <= 0.3)
						compare++;

				}
				for (int c = 0; c < compare; c++) {
					//
					CurrentVoteWeigth += SvmAdaBoost[c];
				}

				if (CurrentVoteWeigth > winnerWeight) {
					winnerWeight = CurrentVoteWeigth;
					maxVote[pred_n] = prediction_table[current][pred_n + 1];
				}
			}
		}
		totalPerformance[i] = performance2(maxVote, testClass, test_size,
				predict_error_th);

		if (i==SVM_COUNT-1) {
			ofstream myfile;
				myfile.open("output_data.csv");
				myfile <<"Actual,Predicted"<<endl;
						for (int t=0;t<SVM_COUNT;t++)
							myfile <<testClass[t]<<","<<maxVote[t]<<endl;
						myfile.close();
		}


		delete[] maxVote;
//TODO HERE
		meanPerformanceAdaBoost[i] = mean(totalPerformance, i + 1);
		printf("Mean Performance %u Ada: %3.2f Raw: %3.2f\n", i,
				meanPerformanceAdaBoost[i], meanPerformanceRaw[i]);

	}

	for (int i = 0; i < returns.count; i++) {
		returns.meanPerformanceAdaBoost[i] = meanPerformanceAdaBoost[i];
		returns.meanPerformanceRaw[i] = meanPerformanceRaw[i];
	}
	return returns;
}

struct fann_train_data *read_from_array(float *din, float *dout,
		unsigned int num_data, unsigned int num_input,
		unsigned int num_output) {
	unsigned int i, j;
	fann_type *data_input, *data_output;
	struct fann_train_data *data = (struct fann_train_data *) malloc(
			sizeof(struct fann_train_data));
	if (data == NULL) {
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	fann_init_error_data((struct fann_error *) data);

	data->num_data = num_data;
	data->num_input = num_input;
	data->num_output = num_output;
	data->input = (double **) calloc(num_data, sizeof(double *));
	if (data->input == NULL) {
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}

	data->output = (double **) calloc(num_data, sizeof(double *));
	if (data->output == NULL) {
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}

	data_input = (double *) calloc(num_input * num_data, sizeof(double));
	if (data_input == NULL) {
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}

	data_output = (double *) calloc(num_output * num_data, sizeof(double));
	if (data_output == NULL) {
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}

	for (i = 0; i != num_data; i++) {
		data->input[i] = data_input;
		data_input += num_input;

		for (j = 0; j != num_input; j++) {
			data->input[i][j] = din[i * num_input + j];
		}

		data->output[i] = data_output;
		data_output += num_output;

		for (j = 0; j != num_output; j++) {
			data->output[i][j] = dout[i * num_output + j];
		}
	}
	return data;
}

fann** do_training_ann(size_t SVM_COUNT, size_t test_size, int col1, int row1,
		float* array1, float* array2, bool classification, bool cascade,
		unsigned int num_layers, unsigned int num_neurons_hidden,
		float desired_error, unsigned int max_epochs,
		unsigned int max_neurons) {
	fann *ann;
	fann** anns;

	anns = Malloc(fann*,SVM_COUNT);

	//TODO PARAMETERS
//	const unsigned int num_layers = 4;
//		const unsigned int num_neurons_hidden = 200;
	//	 float desired_error =  0.0001;
	//const unsigned int max_epochs = 600;
	const unsigned int epochs_between_reports = 100;

	//	unsigned int max_neurons = 200;
	unsigned int neurons_between_reports = 10;

	unsigned int i = 0;

	printf("Creating network.\n");
//Start-----------------------------------

	/*
	 #ifdef OMP
	 int iCPU = omp_get_num_procs();
	 omp_set_num_threads(iCPU);
	 #pragma omp parallel for private (ann)
	 #endif
	 */
	for (size_t svm_machine = 0; svm_machine < SVM_COUNT; svm_machine++) {
		float predicted_total = 0;
		struct fann_train_data *train_data, *test_data;
		size_t trainSize;

		int* numbers = random_number(row1, test_size);

		const size_t trainDataSize = row1 - test_size;
		trainSize = trainDataSize;
		float* testData = new float[test_size * col1 + 1];
		float* testClass = new float[test_size + 1];

		float* trainClass = new float[trainDataSize + 1];

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

				trainClass[trainPos++] = (double) array2[i];	//--debug here
			}
		}

		train_data = read_from_array(trainData, trainClass, row1 - test_size,
				col1, 1);
		test_data = read_from_array(testData, testClass, test_size, col1, 1);
		if (!cascade) {
			ann = fann_create_standard(num_layers, train_data->num_input,
					num_neurons_hidden, num_neurons_hidden / 2,
					train_data->num_output);

			//printf("Training network.\n");
			//fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
			//fann_set_learning_momentum(ann, 0.4f);
			//fann_set_activation_function_layer(ann,FANN_)
			//fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC_STEPWISE);
			//fann_set_activation_function_output(ann, FANN_SIGMOID_STEPWISE);

			/*fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL); */
			fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
			fann_set_activation_function_output(ann, FANN_LINEAR);
			fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
			fann_set_scaling_params(ann, train_data, -1, /* New input minimum */
			1, /* New input maximum */
			0, /* New output minimum */
			10); /* New output maximum */

			fann_scale_train(ann, train_data);

			fann_train_on_data(ann, train_data, max_epochs,
					epochs_between_reports, desired_error);
		} else {
			ann = fann_create_shortcut(2, fann_num_input_train_data(train_data),
					fann_num_output_train_data(train_data));
			enum fann_activationfunc_enum activation;
			fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
			fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
			fann_set_activation_function_output(ann, FANN_LINEAR);
			fann_set_train_error_function(ann, FANN_ERRORFUNC_LINEAR);

			fann_set_scaling_params(ann, train_data, -1, /* New input minimum */
			1, /* New input maximum */
			0, /* New output minimum */
			10); /* New output maximum */

			fann_scale_train(ann, train_data);

			int multi = 0;

			if (!multi) {
				/*steepness = 0.5;*/
				fann_type steepness = 1;
				fann_set_cascade_activation_steepnesses(ann, &steepness, 1);
				/*activation = FANN_SIN_SYMMETRIC;*/

				activation = FANN_SIGMOID_SYMMETRIC;

				fann_set_cascade_activation_functions(ann, &activation, 1);
				fann_set_cascade_num_candidate_groups(ann, 8);
			}

			fann_set_train_stop_function(ann, FANN_STOPFUNC_MSE);
			//	fann_print_parameters(ann);

			fann_cascadetrain_on_data(ann, train_data, max_neurons,
					neurons_between_reports, desired_error);
		}
		//fann_print_connections(ann);
		//	fann_print_parameters(ann);
		printf("Testing network.\n");
		fann_type *calc_out;
		//test_data = //fann_read_train_from_file("test.data");
//float predicted_total=0;
		//fann_reset_MSE(ann);
		for (int i = 0; i < fann_length_train_data(test_data); i++) {
			fann_test(ann, test_data->input[i], test_data->output[i]);
			fann_reset_MSE(ann);
			fann_scale_input(ann, test_data->input[i]);
			calc_out = fann_run(ann, test_data->input[i]);
			fann_descale_output(ann, calc_out);
			float diff = (float) fann_abs(calc_out[0] - test_data->output[i][0]);
#ifdef DEBUG
			printf("Result %f original %f error %f\n",
					calc_out[0], test_data->output[i][0],diff
			);
#endif
			if (classification) {
				if (diff <= 0.5)
					predicted_total++;
				else if (calc_out[0] < 1)
					predicted_total++;
			} else if (diff <= 1.1)
				predicted_total++;
		}

		int test_size = fann_length_train_data(test_data);
		float predicted = (float) (predicted_total / test_size) * 100;
		//printf("MSE error on test data: %f\n", fann_get_MSE(ann));
//	printf("ANN: %d Total_size:%d Performance  %f\%\n",svm_machine,test_size,predicted);

		//printf("Saving network.\n");

		//fann_save(ann, "data.net");

		printf("Cleaning up.\n");

		fann_destroy_train(train_data);
		fann_destroy_train(test_data);

		//fann_destroy(ann);

		anns[svm_machine] = ann;

		printf(" Performance in ann %u= %f\%\n", svm_machine, predicted);

	}

	return anns;
}
;



float descale(float value, float min_scale, float max_scale, float minV,
		float maxV) {
	return min_scale + (max_scale - min_scale) * (value - minV) / (maxV - minV);
}
#ifdef GPU_TEST
#include "GAFeedForwardNN.h"
#include "FeedForwardNN.h"
#include "LearningSet.h"
#include "FeedForwardNNTrainer.h"
#include "ActivationFunctions.h"
#include <string.h>
FeedForwardNN** do_training_ann_gpu(size_t SVM_COUNT, size_t test_size,
		int col1, int row1, float* array1, float* array2, bool classification,
		bool cascade, unsigned int num_layers, unsigned int num_neurons_hidden,
		float desired_error, unsigned int max_epochs,
		unsigned int max_neurons) {

	//activation functions (1=sigm,2=tanh)
	int* functs = new int[num_layers];
	int* layers = new int[num_layers];
	float minV = 0, maxV = 0;

	array1 = scale2(row1, col1, array1, 1, -1, minV, maxV);
//	array2=scale2(1,row1,array2,1,-1,minV,maxV);

	if (num_layers == 3) {
		layers[0] = col1;
		layers[1] = max_neurons;
		layers[2] = 1;
		functs[0] = 2;
		functs[1] = 1;
		functs[2] = 2;

	} else {
		layers[0] = col1;
		layers[1] = max_neurons;
		layers[2] = max_neurons / 2;
		layers[3] = 1;
		functs[0] = 2;
		functs[1] = 1;
		functs[2] = 1;
		functs[3] = 2;
	}

	FeedForwardNN** mynets = Malloc(FeedForwardNN*,SVM_COUNT);

	unsigned int i = 0;

	printf("Creating network.\n");
//Start-----------------------------------

	/*
	 #ifdef OMP
	 int iCPU = omp_get_num_procs();
	 omp_set_num_threads(iCPU);
	 #pragma omp parallel for private (ann)
	 #endif
	 */

	for (size_t svm_machine = 0; svm_machine < SVM_COUNT; svm_machine++) {
		float predicted_total = 0;

		size_t trainSize;

		int* numbers = random_number(row1, test_size);

		const size_t trainDataSize = row1 - test_size;
		trainSize = trainDataSize;
		float* testData = new float[test_size * col1 + 1];
		float* testClass = new float[test_size + 1];

		float* trainClass = new float[trainDataSize + 1];

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

				trainClass[trainPos++] = array2[i];	//--debug here
			}
		}

		trainData = scale2(trainSize, col1, trainData, 1, -1, minV, maxV);
		testData = scale2(test_size, col1, testData, 1, -1, minV, maxV);
//		float* trainBinClass = new float[trainDataSize*3];
//		memset(trainBinClass,NULL,sizeof(float)*trainDataSize*3);
//		for (int i=0;i<trainDataSize;i++) {
//			trainBinClass[i*3+(int)trainClass[i]-1]=1.0;

//		}
		LearningSet trainingSet(trainDataSize, col1, 1, trainData, trainClass);
		LearningSet testSet(test_size, col1, 1, testData, testClass);

		//declare the network with the number of layers
		int layersq = num_layers;

		FeedForwardNN mynet(layersq, layers, functs);

		FeedForwardNNTrainer trainer;
		trainer.selectNet(mynet);
		trainer.selectTrainingSet(trainingSet);
		//trainer.selectTestSet(testSet);

		float param[] = { TRAIN_GPU, ALG_BATCH, desired_error,
				(float) max_epochs, 100, 0.8, 0.0, SHUFFLE_ON, ERROR_TANH };
		trainer.train(9, param);

//		GAFeedForwardNN evo;

		//choose a net to save the best training
//		FeedForwardNN mynet;
		//	evo.selectBestNet(mynet);

		//evo.selectTrainingSet(trainingSet);
		//	evo.selectTestSet(testSet);

		//evolution parameters:
		//popolation
		//generations
		//selection algorithm ROULETTE_WHEEL - TOURNAMENT_SELECTION
		//training for each generated network
		//crossover probability
		//mutation probability
		//number of layers
		//max layer size
//		evo.init(3,5,ROULETTE_WHEEL,2,0.5,0.3,2,100);

		//training parameters:
		//TRAIN_GPU - TRAIN_CPU
		//ALG_BATCH - ALG_BP (batch packpropagation or standard)
		//desired error
		//total epochs
		//epochs between reports

//		float param[]={TRAIN_GPU,ALG_BATCH,0.001,600,100};

		//	evo.selectBestNet(mynet);

//		evo.evolve(5,param,PRINT_MIN);
//		evo.selectBestNet(mynet);
		//	mybest.saveToTxt("../mybestbwc.net");

		float* calc_out = new float[1];
		for (i = 0; i < test_size; i++) {
//float* test = new float[col1];

//memcpy(test,testData+(i*col1),sizeof(float)*col1);
			//float* out =new float[3];
			mynet.compute(testData + (i * col1), calc_out);

			//printf("%f %f %f\n", out[0],out[1],out[2]);

//	calc_out[0]=(float)	mynet.classificate(testData+(i*col1));

//			calc_out[0]=descale(calc_out[0],minV,maxV,0,10);
//			testClass[i]=descale(testClass[i],minV,maxV,0,10);

			float diff = (float) abs(calc_out[0] - testClass[i]);
#ifdef DEBUG
			printf("Result %f original %f error %f\n",
					calc_out[0], testClass[i],diff
			);
#endif
			if (classification) {
				if (diff <= 0.5)
					predicted_total++;
				else if (calc_out[0] < 1)
					predicted_total++;
			} else if (diff <= 1.1)
				predicted_total++;
		}

		//int test_size=fann_length_train_data(test_data);
		float predicted = (float) (predicted_total / test_size) * 100;
		//printf("MSE error on test data: %f\n", fann_get_MSE(ann));
		//	printf("ANN: %d Total_size:%d Performance  %f\%\n",svm_machine,test_size,predicted);

		//printf("Saving network.\n");

		//fann_save(ann, "data.net");

		printf("Cleaning up.\n");

		mynets[svm_machine] = &mynet;

		printf(" Performance in GPU ann %d= %f\%\n", svm_machine, predicted);
		//		mynet->saveToTxt("mynetmushrooms.net");
//trainer.~FeedForwardNNTrainer();
//delete[] testClass;
//delete[] testData;
//delete[] trainClass;
//delete[] trainData;

	}
	return mynets;
}
#endif

svRet analysis_neural(int SVM_COUNT, size_t test_size, int col1, int row1,
		float* array1, float* array2, fann** anns) {
	svRet returns;
	returns.count = SVM_COUNT;
	returns.meanPerformanceAdaBoost = new float[SVM_COUNT];
	returns.meanPerformanceRaw = new float[SVM_COUNT];

	fann *ann;

	//TODO ADA_START
//---------------------Plot means
	float** prediction_table = Malloc(float*,SVM_COUNT);

	//--Test Data
	int* numbers = random_number(row1, test_size, time(NULL));
	//const size_t trainDataSize = row1 - test_size;

	float* testData = new float[test_size * col1 + 1];
	float* testClass = new float[test_size + 1];

	for (size_t i = 0; i < test_size; i++) {

		int number = numbers[i];
		for (int c = 0; c < col1; c++) {
			testData[i * col1 + c] = array1[number * col1 + c];	//--debug here
		}
		testClass[i] = array2[number];		//--debug here

	}

	float meanPerformanceRaw[SVM_COUNT];
	//float meanPerformanceRaw[SVM_COUNT];
	float meanPerformanceAdaBoost[SVM_COUNT];

//AdaBoost
	float Cerror[SVM_COUNT];
	float SampleErrors[test_size];
	double SvmAdaBoost[SVM_COUNT];
	float totalPerformanceS[SVM_COUNT];
	for (size_t i = 0; i < test_size; i++)
		SampleErrors[i] = 1.0 / (float) test_size;
	fann_type *calc_out;
	struct fann_train_data *test_data;
	for (size_t i = 0; i < SVM_COUNT; i++) {

		float* dec_values;
		ann = anns[i];
		//double* maxVote = new double[current+1];
		dec_values = Malloc(float,test_size);
		test_data = read_from_array(testData, testClass, test_size, col1, 1);
		for (size_t i1 = 0; i1 < test_size; i1++) {

			fann_test(ann, test_data->input[i1], test_data->output[i1]);
			fann_reset_MSE(ann);
			fann_scale_input(ann, test_data->input[i1]);
			calc_out = fann_run(ann, test_data->input[i1]);
			fann_descale_output(ann, calc_out);
			dec_values[i1] = calc_out[0];
		}

		prediction_table[i] = dec_values;
/////----------------
		//Total performance raw
		totalPerformanceS[i] = performance(prediction_table[i], testClass,
				test_size, predict_error_th);

		meanPerformanceRaw[i] = mean(totalPerformanceS, i + 1);

		/////////----------------
		for (size_t current = 0; current <= i; current++) {

			//dec_values;
			int* indexes = NULL;
			int ind_length = wrong_compare(prediction_table[current], testClass,
					test_size, &indexes);
			float* Error = new float[ind_length];
			for (int val = 0; val < ind_length; val++) {
				Error[val] = SampleErrors[indexes[val]];
			}
			Cerror[i] = sum(Error, ind_length);

		}

		double w = 1.0 / 3.0 * log((1 - Cerror[i]) / Cerror[i]);
		SvmAdaBoost[i] = w;

		for (size_t t = 0; t < test_size; t++) {

			if (!compare(testClass[t], prediction_table[i][t + 1])) {
				SampleErrors[t] *= pow(2.718282, -w);
			}

		}
//TODO _REMOVE

		for (size_t t = 0; t < test_size; t++) {

			SampleErrors[t] = SampleErrors[t] / (sum(SampleErrors, test_size));

		}

	}

	for (size_t i = 0; i < SVM_COUNT; i++) {
		float totalPerformance[SVM_COUNT];

		float* maxVote = new float[test_size];

		for (size_t pred_n = 0; pred_n < test_size; pred_n++) {
			float winnerWeight = 0;
			for (size_t current = 0; current <= i; current++) {

				int compare = 0;

				float CurrentVoteWeigth = SvmAdaBoost[current];
				double currentVote = prediction_table[current][pred_n + 1];
				//float currentVoteWeight=0;

				for (size_t n = 0; n <= current; n++) {
					double pre_vote = (prediction_table[n][pred_n + 1]);
					if (abs(pre_vote - currentVote) <= 0.3)
						compare++;

				}
				for (int c = 0; c < compare; c++) {
					//
					CurrentVoteWeigth += SvmAdaBoost[c];
				}

				if (CurrentVoteWeigth > winnerWeight) {
					winnerWeight = CurrentVoteWeigth;
					maxVote[pred_n] = prediction_table[current][pred_n + 1];
				}
			}
		}
		totalPerformance[i] = performance2(maxVote, testClass, test_size,
				predict_error_th);
		delete[] maxVote;
//TODO HERE
		meanPerformanceAdaBoost[i] = mean(totalPerformance, i + 1);
		printf("Mean Performance %u Ada: %3.2f Raw: %3.2f\n", i,
				meanPerformanceAdaBoost[i], meanPerformanceRaw[i]);

	}

	for (int i = 0; i < returns.count; i++) {
		returns.meanPerformanceAdaBoost[i] = meanPerformanceAdaBoost[i];
		returns.meanPerformanceRaw[i] = meanPerformanceRaw[i];
	}
	return returns;
}

