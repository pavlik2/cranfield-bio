/*
 * add.c
 *
 *  Created on: 20 Apr 2013
 *      Author: pavel
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
//#include <omp.h>
#include <time.h>
#include "../strtk/strtk.hpp"
#define max(x,y) (((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))
using namespace std;

void wait_for_key ()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)  // every keypress registered, also arrow keys
    cout << endl << "Press any key to continue..." << endl;

    FlushConsoleInputBuffer(GetStdHandle(STD_INPUT_HANDLE));
    _getch();
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
    cout << endl << "Press ENTER to continue..." << endl;

    std::cin.clear();
    std::cin.ignore(std::cin.rdbuf()->in_avail());
    std::cin.get();
#endif
    return;
}



bool isInArray(int num, int* array, int size) {

	for (int i = 0; i < size; i++) {
		if (num == array[i])
			return true;
	}

	return false;

}

double* scale(int row,int col,float* array,
		double max_scale,double min_scale) {
	double* ret_row= new double[col];
	double minV=(double) array[row * col + 0];
	double maxV=(double) array[row * col + 0];
	for (int c = 1; c < col; ++c) {
		minV = min(minV,(double) array[row * col + c]);
		maxV = max(maxV,(double) array[row * col + c]);
												}
	for (int c = 0; c < col; ++c) {
		double value = (double) array[row * col + c];
		if(value == minV)
				value = min_scale;
			else if(value == maxV)
				value = max_scale;
			else
				value = min_scale + (max_scale-min_scale) *
					(value-minV)/
					(maxV-minV);
		ret_row[c]=value;								}

return ret_row;
}

float* scale2(int row1,int col,float* array,
		float max_scale,float min_scale,float& minV,float& maxV) {

for (int i=0;i<row1;i++) {
int row=i;
	//float
	minV= array[row * col + 0];
//	float
	maxV= array[row * col + 0];

	for (int c = 1; c < col; ++c) {
		minV = min(minV, array[row * col + c]);
		maxV = max(maxV, array[row * col + c]);
												}

	for (int c = 0; c < col; ++c) {
		float value =  array[row * col + c];
		if(value == minV)
				value = min_scale;
			else if(value == maxV)
				value = max_scale;
			else
				value = min_scale + (max_scale-min_scale) *
					(value-minV)/
					(maxV-minV);
		array[row * col + c]=value;								}

}
return array;
}

bool compare(double one,double two) {
	if (abs(one - abs(two)) <= 1.1)
		return true;
	else
		return false;
}

float sum(float* array,int length) {
	float ret=0;
	for (int i=0;i<length;i++) {
		ret+=array[i];
	}
	return ret;
}

int wrong_compare(double* array,double* compate_to,int length,int** indexes2) {
int compare=0;
int* indexes = (int*) malloc (sizeof(int) * 1);
	for (int i=0;i<length;i++) {
		if (abs(array[i +1] - abs(compate_to[i])) > 1.1) {
++compare;
indexes = (int*) realloc (indexes,sizeof(int) * compare);
		indexes[compare-1]=i;

		}
		}
	* indexes2=indexes;
		return compare;

}

int wrong_compare(float* array,float* compate_to,int length,int** indexes2) {
int compare=0;
int* indexes = (int*) malloc (sizeof(int) * 1);
	for (int i=0;i<length;i++) {
		if (abs(array[i +1] - abs(compate_to[i])) > 1.1) {
++compare;
indexes = (int*) realloc (indexes,sizeof(int) * compare);
		indexes[compare-1]=i;

		}
		}
	* indexes2=indexes;
		return compare;

}

float mean (float * array, int N)
{
double sum = 0 ;
for (int i = 0; i < N; i++)
sum = sum + array [i];
return sum/N;
} // function calculating mean
//template <class T>

float performance (double* votes,double* actual_values,int test_size,float prediction_factor) {
//performance<-(length(which(maxVote==testCl))/length(testCl))*100
	float predicted_total = 0;
						for (int i = 1; i <= test_size; i++)
							if (abs(actual_values[i - 1] - abs(votes[i])) <= prediction_factor)
								predicted_total += 1.0;

	return (predicted_total / test_size) * 100;
}

float performance (float* votes,float* actual_values,int test_size,float prediction_factor) {
//performance<-(length(which(maxVote==testCl))/length(testCl))*100
	float predicted_total = 0;
						for (int i = 1; i <= test_size; i++)
							if (abs(actual_values[i - 1] - abs(votes[i])) <= prediction_factor)
								predicted_total += 1.0;

	return (predicted_total / test_size) * 100;
}

float performance2 (double* votes,double* actual_values,int test_size,float prediction_factor){
	float predicted_total = 0;
							for (int i = 0; i < test_size; i++)
								if (abs(actual_values[i ] - abs(votes[i])) <= prediction_factor)
									predicted_total += 1.0;

		return (predicted_total / test_size) * 100;
}
float performance2 (float* votes,float* actual_values,int test_size,float prediction_factor){
	float predicted_total = 0;
							for (int i = 0; i < test_size; i++)
								if (abs(actual_values[i ] - abs(votes[i])) <= prediction_factor)
									predicted_total += 1.0;

		return (predicted_total / test_size) * 100;
}
float* array_create(string data,int& col,int& row,int collumnToSave)
{
	float* array_of_csv;
   strtk::token_grid grid(data,data.size(),",");

       size_t max_row = grid.row_count();
        size_t max_col = grid.max_column_count();
      row = max_row-1;
      col=max_col-1;
      if (collumnToSave!=0)
      array_of_csv = new float[max_row];
      else
    	  array_of_csv = new float[max_row*max_col];

if (row>0)
      for (std::size_t row1 = 1; row1 < max_row; row1++)
      {int row=row1-1;
    	  if (collumnToSave!=0) {
    		  std::size_t col1 = collumnToSave;
    		  array_of_csv[row] =  grid.get<float>(row1,col1);
    	  }
    	  else
    	  {
    		  for (size_t col1=1;col1<max_col;col1++) {
    			  int col=col1-1;

        	 array_of_csv[row*(max_col-1)+col] =  grid.get<float>(row1,col1);
    		  }

    	  }
      }
//printf("%f\n",array_of_csv[0]);
return array_of_csv;
}


float* array_create(string data,int& col,int& row)
{

return array_create( data, col, row,0);

}

string open_file(const char* name) {
	string line;
	string line_out="";
	  ifstream myfile (name);
	  if (myfile.is_open())
	  {
	    while ( myfile.good() ) {

	      getline (myfile,line);
	      line_out+=line+"\n";
	     // cout << line << endl;
	    }
	    myfile.close();
	  }

	  else cout << "Unable to open file";

	  return line_out;
   }

bool unique_num(int* num,int size,int numb) {

	for (int i=0;i<size;i++) {
	if (num[i]==numb) return true;
	}

	return false;
}

int* random_number(int size,int test_size,int rand1) {
//	size_t test_size=size/4;
int* numbers = new int [test_size];
srand (time(NULL)*rand1);
//#pragma omp parallel for
for (int var = 0;  var < test_size; ++var) {

int numb = rand() % size ;
while (unique_num(numbers,var,numb)) {
	numb = rand() % size ;
}
	  /* generate secret number between 1 and 10: */
	numbers[var] = numb;
}


	return  numbers; // generates number in the range 1..6



}

int* random_number(int size,int test_size) {
//	size_t test_size=size/4;
int* numbers = new int [test_size];
srand (time(NULL));
//#pragma omp parallel for
for (int var = 0;  var < test_size; ++var) {

int numb = rand() % size ;
while (unique_num(numbers,var,numb)) {
	numb = rand() % size ;
}
	  /* generate secret number between 1 and 10: */
	numbers[var] = numb;
}


	return  numbers; // generates number in the range 1..6



}

void array_transpose(string data,  char* fileName) {

	strtk::token_grid grid(data, data.size(), ",");

	size_t max_row = grid.row_count();
	size_t max_col = grid.max_column_count();
	ofstream myfile;
	const char* tt= ".transposed.csv";
	char* filen=strcat(fileName, tt);
	myfile.open(filen);

	for (size_t col1 = 0; col1 < max_col; col1++) {
		for (std::size_t row1 = 0; row1 < max_row; row1++) {

			if (row1 == max_row - 1)
				myfile << grid.get<string>(row1, col1);
			else

				myfile << grid.get<string>(row1, col1) + ",";
		}
		myfile << endl;
	}
	myfile.close();
exit(0);
}

