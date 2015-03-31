/*
 * add.h
 *
 *  Created on: 22 Apr 2013
 *      Author: pavel
 */

#ifndef ADD_H_
#define ADD_H_
using namespace std;
#include <string>
void wait_for_key ();
float* array_create(string data,int& col,int& row,int collumnToSave);
float* array_create(string data,int& col,int& row);
string open_file(const char* name);
int* random_number(int size,int test_size,int rand);
int* random_number(int size,int test_size);
float mean (float * array, int N);
float performance (double* votes,double* actual_values,int test_size,float prediction_factor);
float performance2 (double* votes,double* actual_values,int test_size,float prediction_factor);
int wrong_compare(double* array,double* compate_to,int length,int** indexes2);
float performance (float* votes,float* actual_values,int test_size,float prediction_factor);
float performance2 (float* votes,float* actual_values,int test_size,float prediction_factor);
int wrong_compare(float* array,float* compate_to,int length,int** indexes2);
float sum(float* array,int length);
bool compare(double one,double two);
double* scale(int row,int col,float* array,
		double max_scale,double min_scale);
float* scale2(int row1,int col,float* array,
		float max_scale,float min_scale,float& minV,float& maxV);
//double* min(float* array,);
bool isInArray(int num, int* array, int size);
void array_transpose(string data,  char* fileName);
#endif /* ADD_H_ */
