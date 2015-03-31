/*
 * cpp_bio_light.cpp
 *
 *  Created on: 22 Apr 2013
 *      Author: pavel
 */


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "main/add.cpp"

using namespace std;

int main(void) {

string data1= 	open_file("enose");
string data2= 	open_file("bacterial");
printf("done read files\n");
int col1,row1;
int* array1;
cout << data1;

//array_create<int>(data1,array1,col1,row1);


int col2,row2;
float* array2;

//array_create<float>(data2,array2,col2,row2);
printf("done2\n");
float* class_s = new float[row1];

//for (int i=0;i<row1;i++)
//	class_s[i]=array2[i*col1];

size_t test_size=row1/4;
//--------------------------------------------------
printf("Done\n");




}

