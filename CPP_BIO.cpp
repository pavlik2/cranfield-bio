//============================================================================
// Name        : CPP_BIO.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <add.h>
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
#include <doublefann.h>
#include <FeedForwardNN.h>
#include "gnuplot_i.hpp"
#include <boost/program_options/cmdline.hpp>
#include "boost/program_options.hpp"
#include <core.h>
#include <gpu_core.h>
#include <core_functions.h>
//size_t SVM_COUNT = 1;
int max_iter = 1; 	//TODO - Declare
#define ADA_BOOST
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
//#define OMP

//#define TEST_SPEED
#ifdef DEBUG
#define DEBUG_SVM
#define PRINT_ALL
#endif
using namespace std;
bool ann = false;
bool show_plot = false;
struct svm_node *x_space;
// This program uses
int cross_validation;
int nr_fold;

void gnu_plot(svRet perf);
bool load = false;
bool train_test = false;
int main(int argc, char* argv[]) {
	namespace po = boost::program_options;
//Options with command line
	po::options_description desc("Allowed options");
	desc.add_options()("help", "produce help message")("C", po::value<float>(),
			"C level")("svm", po::value<int>(), "svm=svm_count")("g",
			po::value<float>(), "1")("Cstep", po::value<float>(), "2")("gStep",
			po::value<float>(), "3")("iter", po::value<int>(), "4")("data",
			po::value<string>(), "Data for class")("data1", po::value<string>(),
			"Enose data")("classification", po::value<bool>(), "true/false")(
			"show-graph", po::value<bool>(), "true/false")("ann",
			po::value<bool>(), "true/false")("Cstart", po::value<float>(),
			"default -1")("cascade", po::value<bool>(), "default false")(
			"gStart", po::value<float>(), "default -0.1")("num_layers",
			po::value<unsigned int>(), "default 4")("num_neurons_hidden",
			po::value<unsigned int>(), "default 200")("max_epochs",
			po::value<unsigned int>(), "default 300")("max_neurons",
			po::value<unsigned int>(), "default 100")("desired_error",
			po::value<float>(), "default 0.0001")("gpu", po::value<bool>(),
			"true/false")("load", po::value<bool>(), "true/false")("train_test",
			po::value<bool>(), "true/false");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		cout << desc << "\n";
		return 1;
	}
//Initial parameters
	float C = 50.0, g = 10.0, Cstep = 1.0, gStep = 0.1;

	size_t SVM_COUNT = 1;

	int iter = 1;

	if (vm.count("C")) {
		C = vm["C"].as<float>();
	}
//G parameter
	if (vm.count("g")) {
		g = vm["g"].as<float>();

	}

	if (vm.count("Cstep")) {
		Cstep = vm["Cstep"].as<float>();

	}

	if (vm.count("gStep")) {
		gStep = vm["gStep"].as<float>();

	}

	if (vm.count("iter")) {
		iter = vm["iter"].as<int>();

	}

	if (vm.count("svm")) {
		SVM_COUNT = vm["svm"].as<int>();
	}

	bool cascade = false;
	if (vm.count("cascade")) {
		cascade = vm["cascade"].as<bool>();
	}
//File options
	const char *file1, *file2;
	file1 = "EnoseAllSamples.csv";
	file2 = "SensoryAllSamples.csv";

	if (vm.count("data")) {
		file1 = vm["data"].as<string>().c_str();

	}

	if (vm.count("data1")) {
		file2 = vm["data1"].as<string>().c_str();

	}
	bool classification = true;
	if (vm.count("classification")) {
		classification = vm["classification"].as<bool>();

	}
//parameter for neural network
	unsigned int num_layers = 4;
	if (vm.count("num_layers")) {
		num_layers = vm["num_layers"].as<unsigned int>();

	}
	//parameter for hidden neurons count
	unsigned int num_neurons_hidden = 200;
	if (vm.count("num_neurons_hidden")) {
		num_neurons_hidden = vm["num_neurons_hidden"].as<unsigned int>();

	}
	float desired_error = 0.0001;
	if (vm.count("desired_error")) {
		desired_error = vm["desired_error"].as<float>();

	}
	unsigned int max_epochs = 300;

	if (vm.count("max_epochs")) {
		max_epochs = vm["max_epochs"].as<unsigned int>();

	}

	unsigned int max_neurons = 200;

	if (vm.count("max_neurons")) {
		max_neurons = vm["max_neurons"].as<unsigned int>();

	}

	if (vm.count("ann")) {
		ann = vm["ann"].as<bool>();
	}

	bool gpu = false;
	if (vm.count("gpu")) {
		gpu = vm["gpu"].as<bool>();

	}

	if (vm.count("load")) {
		load = vm["load"].as<bool>();

	}

	if (vm.count("train_test")) {
		train_test = vm["train_test"].as<bool>();

	}

	if (vm.count("show-graph")) {
		show_plot = vm["show-graph"].as<bool>();

	}
	float Cstart = 1, gStart = 0.1;
	if (vm.count("Cstart")) {
		Cstart = vm["Cstart"].as<float>();
	}
	if (vm.count("gStart")) {
		gStart = vm["gStart"].as<float>();
	}

	string data1 = open_file(file1);
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

	float* testData;

	double* testClass;

	double* trainClass;

	float* trainData;

	size_t test_size = row1 / 4;

	divide_sets(array1, array2, row1, col1, test_size, trainData, trainClass,
			testData, testClass);



//--------------------------------------------------
	svRet perf;
#ifdef DEBUG
	printf("Done\n");
#endif

#ifdef GPU_TEST
	if (!ann)
	gpu_process(Cstart, gStart, C, Cstep, g, gStep,
			SVM_COUNT, iter, test_size, col1, row1, array1, array2,
			classification);
	else
	do_training_ann_gpu(SVM_COUNT, test_size, col1, row1, array1,
			array2, classification, cascade,num_layers,num_neurons_hidden,desired_error,
			max_epochs,max_neurons);

#else

	if (ann) {
		//Artificial neural networks
		fann** anns = do_training_ann(SVM_COUNT, test_size, col1, row1, array1,
				array2, classification, cascade, num_layers, num_neurons_hidden,
				desired_error, max_epochs, max_neurons);
		//Svm analysis using neural networks
		perf = analysis_neural(SVM_COUNT, test_size, col1, row1, array1, array2,
				anns);
	} else {
		svm_model** models;
		double* array2d = new double[row1];
					for (int i = 0; i < row1; i++)
						array2d[i] = array2[i];
		if (train_test) {
			if (!load)
				models = do_training(Cstart, gStart, C, Cstep, g, gStep,
						SVM_COUNT, iter, col1, row1 - test_size, trainData,
						trainClass, classification, gpu);
			else
				models = load_SVM(SVM_COUNT);

		} else {

			if (!load)
				models = do_training(Cstart, gStart, C, Cstep, g, gStep,
						SVM_COUNT, iter, col1, row1, array1, array2d,
						classification, gpu);
			else
				models = load_SVM(SVM_COUNT);
		}

		//Do Adaboost analysis

		if (!load) {
			save_SVM(SVM_COUNT, models);
			perf = analysis(SVM_COUNT, test_size, col1,  models, testData,
							testClass,classification);
		} else
			{
				testData=array1;
				testClass=array2d;
				perf = analysis(SVM_COUNT, row1, col1,  models, testData,
											testClass,classification);
			}


	}
#endif

#ifndef DEBUG
	gnu_plot(perf);

	return 0;
#endif
}

void gnu_plot(svRet perf) {
	//gnuplot graph
	std::vector<double> x, y;
	for (int i = 0; i < perf.count; i++) {
		x.push_back(perf.meanPerformanceAdaBoost[i]);
		y.push_back(perf.meanPerformanceRaw[i]);
	}

	Gnuplot g1("lines");
	if (ann)
		g1.set_title("Ensemble-Based Classifiers Prediction ANN");
	else
		g1.set_title("Ensemble-Based Classifiers Prediction SVM");
	g1.set_smooth("bezier").plot_x(y, "Majority Voting");

	g1.set_cbrange(60.0, 100.0);

	g1.set_samples(1000);

	g1.set_smooth("bezier").plot_x(x, "AdaBoost");

	g1.savetops("plot");

	g1.set_xautoscale().replot();	 // window output

	g1.savetops("plot.ps");
	g1.showonscreen();
	g1.savetops("plot.ps");

	if (show_plot)
		wait_for_key();
	g1 << "q";

	system("sleep 1");
	system(
			"convert -rotate 90 -alpha deactivate -resize 1024x768 -antialias plot.ps plot.png");
}
