/*
 * gpu.h
 *
 *  Created on: 8 Jul 2013
 *      Author: pavel
 */

#ifndef GPU_H_
#define GPU_H_
typedef float Qfloat;
typedef signed char schar;
#ifdef GPU_TEST


struct svm_p{
	float C;
	float gamma;
};

extern "C"

#ifdef FLOAT1g
float
#else
double
#endif
* RunGPU(int l,
#ifdef FLOAT1g
float
#else
double
#endif
*trainClass,float* train,int dim,  float* svm_train,
		int len,svm_p* array_param,int param_length,bool isClassification);
#else
void initGPU(int l,int dim);
void freeGPU();
//extern "C" double* RunGPU( float* train,int l,int dim,double* py,
	//	float* test,int len);
#ifdef FLOAT1g
void RunGPUKERNEL(const svm_problem& prob,schar *y,
		float gamma,float* data, float* QD) ;
#else
void RunGPUKERNEL(const svm_problem& prob,schar *y,
		double gamma,float* data, double* QD) ;
#endif
#endif
#endif /* GPU_H_ */
