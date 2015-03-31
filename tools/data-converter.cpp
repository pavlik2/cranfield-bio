/*
 * data-converter.cpp
 *
 *  Created on: 12 Aug 2013
 *      Author: pavel
 */




/*
 * data-converter.cpp
 *
 *  Created on: 30 Jul 2013
 *      Author: pavel
 */

/*
 * cpp_bio_light.cpp
 *
 *  Created on: 22 Apr 2013
 *      Author: pavel
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "add.h"
#include <iostream>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <utility>
#include <string>
#include <vector>

#include <utility>
#include <math.h>
#include <omp.h>

#include "strtk/strtk.hpp"
#include <time.h>

#define SVM_COUNT 1
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define OMP
#define GPUSVM
//#define TEST_SPEED
#ifdef DEBUG
#define DEBUG_SVM
#define PRINT_ALL
#endif
using namespace std;
float maxPerformance = 0;
template<typename T> string tostr(const T& t) {
	ostringstream os;
	os << t;
	return os.str();
}


int main(int argc, char* argv[]) {
	srand(time(0));

	char *file1, *file2;char* transponse = new char;

	if (argc >= 3) {
		file1 = argv[1];
		file2 = argv[2];
		transponse = argv[3];
	} else {
		file1 = "beef_fillets_raman.csv";
		//file2 = "BacterialAllSamples.csv";
		file2 = "beef_fillets_raman_sensory.csv";
	}
	string data1 = open_file(file1);
bool neural=false,neural2=false;
char* neur = argv[4];
if (argc >= 4 && neur[0]=='n') neural=true;
if (argc >= 4 && neur[0]=='c') {neural2=true; neural=true;}
	bool transponse1 = false;
	if (transponse[0]=='1')

		transponse1 = true;

	if (!strcmp(transponse,"transpose")) {
		array_transpose(data1,file1);
	}

#ifdef DEBUG
	cout << data1 << endl;
#endif
	int col1 = 0, row1 = 0;
	float* array1 = array_create(data1, col1, row1);
#ifdef DEBUG
	printf("open second file\n");
#endif
	int col2, row2;

	string data2 = open_file(file2);
	float* array2 = array_create(data2, col2, row2, 1);
#ifdef DEBUG

	if (!array2)
	exit(0);

	for (int i = 0; i < row2; i++)
	printf("done2 %f\n", array2[i]);
#endif

	size_t test_size = row1 / 4;

//--------------------------------------------------

#ifdef DEBUG
	printf("Done\n");
#endif
	float modelPerformance = 0.0;
	printf("%s", transponse);
	if (transponse1)
		test_size = col1 / 4;
	//TODO: Declare in header
	int* numbers = random_number(row1, test_size);

	size_t trainDataSize = col1 - test_size;
	if (transponse1) {

		printf("\n Transponsed output data\n");
		numbers = random_number(col1, test_size);
		trainDataSize = col1 - test_size;
	}

	double* testClass = new double[test_size + 1];

	ofstream myfile;
	myfile.open("test.data");

	if (neural)
		myfile<<test_size<<" "<<col1<<" "<<1<<endl;

	if (!transponse1)
		for (size_t i = 0; i < test_size; i++) {
			int number = numbers[i];
			testClass[i] = (double) array2[number];
			if (!neural)
			myfile << array2[number] << " ";

			for (int c = 0; c < col1; c++) {
			if (!neural)
				myfile << c + 1 << ":" << array1[number * col1 + c] << " ";	//--debug here
			else
							myfile <<array1[number * col1 + c] << " ";	//--debug here
			}
		if(!neural2)	myfile << endl;
			if (neural) myfile << array2[number] << endl;
		}
	else {
		test_size = col1 / 4;
		for (size_t i = 0; i < test_size; i++) {
			int number = numbers[i];
			testClass[i] = (double) array2[number];
			if (!neural)
			myfile << array2[number] << " ";

			for (int c = 0; c < row1; c++) {
				if (!neural)	myfile << c + 1 << ":" << array1[c * col1 + number] << " ";	//--debug here
				else
											myfile <<array1[c * col1 + number] << " ";	//--debug here
			}
			if(!neural2) myfile << endl;
			if (neural) myfile << array2[number] << endl;
		}
	}
	myfile.close();
	int trainPos = 0;

	myfile.open("train.data");
	if (neural) myfile <<row1-test_size<<" "<<col1<<" "<<1<<endl;
	if (!transponse1)
		for (int i = 0; i < row1; i++) {

			if (!isInArray(i, numbers, test_size)) {
				if (!neural)	myfile << array2[i] << " ";

				for (int c = 0; c < col1; ++c) {
						if(!neural)
					myfile << c + 1 << ":" << array1[i * col1 + c] << " ";//---CHECK HERE-debug here
					else
							myfile <<array1[i * col1 + c] << " ";	//--debug here
				}
				if(!neural2) myfile << endl;
				if (neural) myfile << array2[i] << endl;
			}
		}
	else
		for (int c = 0; c < col1; ++c) {

			if (!isInArray(c, numbers, test_size)) {
				if (!neural)
				myfile << array2[c] << " ";

				for (int i = 0; i < row1; i++) {
					if(!neural)
					myfile << i + 1 << ":" << array1[i * col1 + c] << " ";//---CHECK HERE-debug here
					else
							myfile <<array1[i * col1 + c] << " ";	//--debug here
				}
				if(!neural2) myfile << endl;
				if (neural) myfile << array2[c] << endl;
			}

		}

	myfile.close();
}
