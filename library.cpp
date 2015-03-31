//#include <iostream>
#include <stdio.h>
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
//#include "svm_function.h"
#include "main/add.cpp"
#include "libsvm-dense-3.17/svm.h"
#include <omp.h>
//#define OMP
void string_stdout(const char *s) {
}
;
bool lim=true;

float predict_error_th = 1.1;
svm_model** do_training(float Cstart, float gStart, float C, float Cstep,
		float gamma, float gammaStep, int SVM_COUNT, int iterations1,
		size_t test_size, int col1, int row1, float* array1, float* array2,
		bool classiffication,bool gpu) {

	//bool classiffication=false;

	int max_iter = iterations1;

	svm_model** models;
	svm_model *model;

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
				svm_node* svm_n;
				size_t trainSize;
				for (int iterations = 0; iterations < iterations1;
						iterations++) {
					srand(time(NULL));
					svm_parameter param;		// set by parse_command_line

					int* numbers = random_number(row1, test_size,
							(iterations + 1));
					const size_t trainDataSize = row1 - test_size;
					trainSize = trainDataSize;
					float* testData = new float[test_size * col1 + 1];
					float* testClass = new float[test_size + 1];


double

* trainClass = new
double

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
								trainData[trainPos * col1 + c] = array1[i * col1
										+ c];		//---CHECK HERE-debug here
							}

							trainClass[trainPos++] = (double) array2[i];//--debug here
						}
					}

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

					model = svm_train(&prob, &param);






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
					delete[] numbers;
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
		printf("Best Performance in svm %d= %f\%\n", svm_machine,
				modelPerformance);
	}
#endif

	return models;
}
;


svRet analysis(int SVM_COUNT, size_t test_size, int col1, int row1,
		float* array1, float* array2, svm_model** models) {
	svRet returns;
	returns.count = SVM_COUNT;
	returns.meanPerformanceAdaBoost = new float[SVM_COUNT];
	returns.meanPerformanceRaw = new float[SVM_COUNT];

	svm_model *model;

	//TODO ADA_START
//---------------------Plot means
	double** prediction_table = Malloc(double*,SVM_COUNT);

	//--Test Data
	int* numbers = random_number(row1, test_size, time(NULL));
	//const size_t trainDataSize = row1 - test_size;

	float* testData = new float[test_size * col1 + 1];
	double* testClass = new double[test_size + 1];

	for (size_t i = 0; i < test_size; i++) {
		int number = numbers[i];
		for (int c = 0; c < col1; c++) {
			testData[i * col1 + c] = array1[number * col1 + c];	//--debug here
		}
		testClass[i] = (double) array2[number];		//--debug here

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

	for (size_t i = 0; i < SVM_COUNT; i++) {
		//float totalPerformance[SVM_COUNT];
		double* dec_values;		//= Malloc(double,test_size + 1);
		model = models[i];
		//double* maxVote = new double[current+1];
		dec_values = Malloc(double,test_size + 1);
		svm_node* svm_t = new svm_node;
		for (size_t i = 0; i < test_size; i++) {

			svm_t->dim = col1;



						svm_t->values =							//[c] =
								scale(i, col1, testData, 1.0, -1.0);

			//		(double) testData[i * col1 + c];//---CHECK HERE-debug here
#ifndef GPU_TEST
			dec_values[i + 1] = svm_predict(model, svm_t);
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

				for (int n = 0; n <= current; n++) {
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
		printf("Mean Performance %d Ada: %3.2f Raw: %3.2f\n", i,
				meanPerformanceAdaBoost[i], meanPerformanceRaw[i]);

	}

	for (int i = 0; i < returns.count; i++) {
		returns.meanPerformanceAdaBoost[i] = meanPerformanceAdaBoost[i];
		returns.meanPerformanceRaw[i] = meanPerformanceRaw[i];
	}
	return returns;
}
#include <R.h>
void trs(int col1,int row1,double* array1d,double* array2d,double* out) {
	float* array1= Malloc(float,row1*col1);
	for (int i=0;i<row1*col1;i++) {array1[i]=(float) array1d[i];
//	printf("%f\n",array1[i]);
	}
	float* array2= Malloc(float,row1);
		for (int i=0;i<row1;i++) array2[i]=(float) array2d[i];

//out[0]=array1[0];
	//return array1;
}

extern "C" {
void do_training1(int* col1,int* row1,double* array1d,double* array2d,double* out,double* out1) {


	int total=row1[0]*col1[0];
	//printf("model2 %d",total);
	out[0]=total;
//	 trs( col1[0], row1[0], array1d, array2d,out);
	 float* array1= Malloc(float,total);
	 	for (int i=0;i<total;i++) {
	 		array1[i]=(float) array1d[i];
	 	//printf("%f\n",array1[i]);
	 	}
	 	float* array2= Malloc(float,row1[0]);
	 			for (int i=0;i<row1[0];i++) array2[i]=(float) array2d[i];
//out[0]=3;
//float* array2= new float[col1*row1];
//for (int i=0;i<row1;i++) array1[i]=array1d[i];
//	for (int i=0;i<row1*col1;i++) array2[i]=array2d[i];
//
	svm_model** models = do_training(1, 0.1, 50, 1, 10, 0.1,
								200, 1, 44, col1[0], row1[0], array1, array2,
								true,false);

	svRet	perf = analysis(200, 44, col1[0], row1[0], array1, array2,
							models);

for (int i=0;i<perf.count;i++) {
	out[i]=perf.meanPerformanceRaw[i];
	out1[i]=perf.meanPerformanceAdaBoost[i];

}
}}
