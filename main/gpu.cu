#include <stdio.h>

#include <helper_cuda.h>

#include <stdlib.h>

#include <vector>
#include <algorithm>
#define _DENSE_REP

/*
 * container.hpp
 *
 *  Created on: 22 Jul 2013
 *      Author: pavel
 */

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/////////////////////////////////////////////////////////////////////////////
//
// Container parent class.
//
////////////////////////////////////////////////////////////////////////////
struct svm_node {
	int dim;
	
#ifdef FLOAT1g
float
#else
double
#endif

 *values;
};

struct svm_problem {
	int l;
	
#ifdef FLOAT1g
float
#else
double
#endif

 *y;
	struct svm_node *x;
};

enum {
	C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
};
/* svm_type */
enum {
	LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED
};
/* kernel_type */

struct svm_parameter {
	int svm_type;
	int kernel_type;
	int degree; /* for poly */
	
#ifdef FLOAT1g
float
#else
double
#endif

 gamma; /* for poly/rbf/sigmoid */
	
#ifdef FLOAT1g
float
#else
double
#endif

 coef0; /* for poly/sigmoid */

	/* these are for training only */
	
#ifdef FLOAT1g
float
#else
double
#endif

 cache_size; /* in MB */
	
#ifdef FLOAT1g
float
#else
double
#endif

 eps; /* stopping criteria */
	
#ifdef FLOAT1g
float
#else
double
#endif

 C; /* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight; /* for C_SVC */
	int *weight_label; /* for C_SVC */
	
#ifdef FLOAT1g
float
#else
double
#endif

* weight; /* for C_SVC */
	
#ifdef FLOAT1g
float
#else
double
#endif

 nu; /* for NU_SVC, ONE_CLASS, and NU_SVR */
	
#ifdef FLOAT1g
float
#else
double
#endif

 p; /* for EPSILON_SVR */
	int shrinking; /* use the shrinking heuristics */
	int probability; /* do probability estimates */
};

//
// svm_model
//
struct svm_model {
	struct svm_parameter param; /* parameter */
	int nr_class; /* number of classes, = 2 in regression/one class svm */
	int l; /* total #SV */

	struct svm_node *SV; /* SVs (SV[l]) */

	
#ifdef FLOAT1g
float
#else
double
#endif

 **sv_coef; /* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	
#ifdef FLOAT1g
float
#else
double
#endif

 *rho; /* constants in decision functions (rho[k*(k-1)/2]) */
	
#ifdef FLOAT1g
float
#else
double
#endif

 *probA; /* pariwise probability information */
	
#ifdef FLOAT1g
float
#else
double
#endif

 *probB;
	int *sv_indices; /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

	/* for classification only */

	int *label; /* label of each class (label[k]) */
	int *nSV; /* number of SVs for each class (nSV[k]) */
	/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
	/* XXX */
	int free_sv; /* 1 if svm_model is created by svm_load_model*/
	/* 0 if svm_model is created by svm_train */
};
__device__ 
#ifdef FLOAT1g
float
#else
double
#endif

 svm_predict(const svm_model *model, const svm_node *x);
__device__ void svm_free_model_content(svm_model* model_ptr) {
	if (model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
#ifdef _DENSE_REP
		for (int i = 0; i < model_ptr->l; i++)
			free(model_ptr->SV[i].values);
#else
	free((void *)(model_ptr->SV[0]));
#endif
	if (model_ptr->sv_coef) {
		for (int i = 0; i < model_ptr->nr_class - 1; i++)
			free(model_ptr->sv_coef[i]);
	}

	free(model_ptr->SV);
	model_ptr->SV = NULL;

	free(model_ptr->sv_coef);
	model_ptr->sv_coef = NULL;

	free(model_ptr->rho);
	model_ptr->rho = NULL;

	free(model_ptr->label);
	model_ptr->label = NULL;

	free(model_ptr->probA);
	model_ptr->probA = NULL;

	free(model_ptr->probB);
	model_ptr->probB = NULL;

	free(model_ptr->sv_indices);
	model_ptr->sv_indices = NULL;

	free(model_ptr->nSV);
	model_ptr->nSV = NULL;
}

__device__ void svm_free_and_destroy_model(svm_model** model_ptr_ptr) {
	if (model_ptr_ptr != NULL && *model_ptr_ptr != NULL) {
		svm_free_model_content(*model_ptr_ptr);
		free(*model_ptr_ptr);
		*model_ptr_ptr = NULL;
	}
};

__device__ 
#ifdef FLOAT1g
float
#else
double
#endif

 svm_predict(const svm_model *model, const svm_node *x);
typedef float Qfloat;
typedef signed char schar;
#ifndef min
template<class T> __device__  static inline T min(T x, T y) {
	return (x < y) ? x : y;
}
#endif
#ifndef max
template<class T> __device__  static inline T max(T x, T y) {
	return (x > y) ? x : y;
}
#endif
template<class T> __device__ static inline void swap(T& x, T& y) {
	T t = x;
	x = y;
	y = t;
}
template<class S, class T> __device__ static inline void clone(T*& dst, S* src,
		int n) {
	dst = new T[n];
	memcpy((void *) dst, (void *) src, sizeof(T) * n);
}

#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class QMatrix {
public:
	__device__	 virtual Qfloat *get_Q(int column, int len) const = 0;
	__device__ virtual 
#ifdef FLOAT1g
float
#else
double
#endif

 *get_QD() const = 0;

	__device__ virtual ~QMatrix() {
	}
};

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//

__device__ static 
#ifdef FLOAT1g
float
#else
double
#endif

 k_function(const svm_node *x, const svm_node *y,
		const svm_parameter& param) {
	
#ifdef FLOAT1g
float
#else
double
#endif

 sum = 0;

	int dim = min(x->dim, y->dim), i;
	for (i = 0; i < dim; i++) {
		
#ifdef FLOAT1g
float
#else
double
#endif

 d = x->values[i] - y->values[i];
		sum += d * d;
	}
	for (; i < x->dim; i++)
		sum += x->values[i] * x->values[i];
	for (; i < y->dim; i++)
		sum += y->values[i] * y->values[i];
	return exp(-param.gamma * sum);

}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
public:
	__device__ Solver() {
	}
	;
	__device__ virtual ~Solver() {
	}
	;

	struct SolutionInfo {
		
#ifdef FLOAT1g
float
#else
double
#endif

 obj;
		
#ifdef FLOAT1g
float
#else
double
#endif

 rho;
		
#ifdef FLOAT1g
float
#else
double
#endif

 upper_bound_p;
		
#ifdef FLOAT1g
float
#else
double
#endif

 upper_bound_n;
		
#ifdef FLOAT1g
float
#else
double
#endif

 r;	// for Solver_NU
	};

	__device__ void Solve(int l, const QMatrix& Q, const 
#ifdef FLOAT1g
float
#else
double
#endif

 *p_,
			const schar *y_, 
#ifdef FLOAT1g
float
#else
double
#endif

 *alpha_, 
#ifdef FLOAT1g
float
#else
double
#endif

 Cp, 
#ifdef FLOAT1g
float
#else
double
#endif

 Cn, 
#ifdef FLOAT1g
float
#else
double
#endif

 eps,
			SolutionInfo* si, int shrinking);
protected:
	int active_size;
	schar *y;
	
#ifdef FLOAT1g
float
#else
double
#endif

 *G;		// gradient of objective function
	enum {
		LOWER_BOUND, UPPER_BOUND, FREE
	};
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	
#ifdef FLOAT1g
float
#else
double
#endif

 *alpha;
	const QMatrix *Q;
	const 
#ifdef FLOAT1g
float
#else
double
#endif

 *QD;
	
#ifdef FLOAT1g
float
#else
double
#endif

 eps;
	
#ifdef FLOAT1g
float
#else
double
#endif

 Cp, Cn;
	
#ifdef FLOAT1g
float
#else
double
#endif

 *p;
	int *active_set;
	
#ifdef FLOAT1g
float
#else
double
#endif

 *G_bar;		// gradient, if we treat free variables as 0
	int l;
	bool unshrink;	// XXX

	__device__ 
#ifdef FLOAT1g
float
#else
double
#endif

 get_C(int i) {
		return (y[i] > 0) ? Cp : Cn;
	}
	__device__ void update_alpha_status(int i) {
		if (alpha[i] >= get_C(i))
			alpha_status[i] = UPPER_BOUND;
		else if (alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else
			alpha_status[i] = FREE;
	}
	__device__ bool is_upper_bound(int i) {
		return alpha_status[i] == UPPER_BOUND;
	}
	__device__ bool is_lower_bound(int i) {
		return alpha_status[i] == LOWER_BOUND;
	}
	__device__ bool is_free(int i) {
		return alpha_status[i] == FREE;
	}

	__device__ void reconstruct_gradient();
	__device__ virtual int select_working_set(int &i, int &j);
	__device__ virtual 
#ifdef FLOAT1g
float
#else
double
#endif

 calculate_rho();

private:
	__device__ bool be_shrunk(int i, 
#ifdef FLOAT1g
float
#else
double
#endif

 Gmax1, 
#ifdef FLOAT1g
float
#else
double
#endif

 Gmax2);
};

__device__ void Solver::reconstruct_gradient() {
	// reconstruct inactive elements of G from G_bar and free variables

	if (active_size == l)
		return;

	int i, j;
	int nr_free = 0;

	for (j = active_size; j < l; j++)
		G[j] = G_bar[j] + p[j];

	for (j = 0; j < active_size; j++)
		if (is_free(j))
			nr_free++;

//	if (2 * nr_free < active_size)
//		//info("\nWARNING: using -h 0 may be faster\n");

	if (nr_free * l > 2 * active_size * (l - active_size)) {
		for (i = active_size; i < l; i++) {
			const Qfloat *Q_i = Q->get_Q(i, active_size);
			for (j = 0; j < active_size; j++)
				if (is_free(j))
					G[i] += alpha[j] * Q_i[j];
		}
	} else {
		for (i = 0; i < active_size; i++)
			if (is_free(i)) {
				const Qfloat *Q_i = Q->get_Q(i, l);
				
#ifdef FLOAT1g
float
#else
double
#endif

 alpha_i = alpha[i];
				for (j = active_size; j < l; j++)
					G[j] += alpha_i * Q_i[j];
			}
	}
}

__device__ void Solver::Solve(int l, const QMatrix& Q, const 
#ifdef FLOAT1g
float
#else
double
#endif

 *p_,
		const schar *y_, 
#ifdef FLOAT1g
float
#else
double
#endif

 *alpha_, 
#ifdef FLOAT1g
float
#else
double
#endif

 Cp, 
#ifdef FLOAT1g
float
#else
double
#endif

 Cn, 
#ifdef FLOAT1g
float
#else
double
#endif

 eps,
		SolutionInfo* si, int shrinking) {
	this->l = l;
	this->Q = &Q;
	QD = Q.get_QD();
	clone(p, p_, l);
	clone(y, y_, l);
	clone(alpha, alpha_, l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	unshrink = false;

	// initialize alpha_status
	{
		alpha_status = new char[l];
		for (int i = 0; i < l; i++)
			update_alpha_status(i);
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for (int i = 0; i < l; i++)
			active_set[i] = i;
		active_size = l;
	}

	// initialize gradient
	{
		G = new 
#ifdef FLOAT1g
float
#else
double
#endif

[l];
		G_bar = new 
#ifdef FLOAT1g
float
#else
double
#endif

[l];
		int i;
		for (i = 0; i < l; i++) {
			G[i] = p[i];
			G_bar[i] = 0;
		}
		for (i = 0; i < l; i++)
			if (!is_lower_bound(i)) {
				const Qfloat *Q_i = Q.get_Q(i, l);
				
#ifdef FLOAT1g
float
#else
double
#endif

 alpha_i = alpha[i];
				int j;
				for (j = 0; j < l; j++)
					G[j] += alpha_i * Q_i[j];
				if (is_upper_bound(i))
					for (j = 0; j < l; j++)
						G_bar[j] += get_C(i) * Q_i[j];
			}
	}

	// optimization step

	int iter = 0;
	int max_iter = max(10000000, l > INT_MAX / 100 ? INT_MAX : 100 * l);
	int counter = min(l, 1000) + 1;

	while (iter < max_iter) {
		// show progress and do shrinking

		if (--counter == 0) {
			counter = min(l, 1000);

			//info(".");
		}

		int i, j;
		if (select_working_set(i, j) != 0) {
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = l;
			//info("*");
			if (select_working_set(i, j) != 0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}

		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully

		const Qfloat *Q_i = Q.get_Q(i, active_size);
		const Qfloat *Q_j = Q.get_Q(j, active_size);

		
#ifdef FLOAT1g
float
#else
double
#endif

 C_i = get_C(i);
		
#ifdef FLOAT1g
float
#else
double
#endif

 C_j = get_C(j);

		
#ifdef FLOAT1g
float
#else
double
#endif

 old_alpha_i = alpha[i];
		
#ifdef FLOAT1g
float
#else
double
#endif

 old_alpha_j = alpha[j];

		if (y[i] != y[j]) {
			
#ifdef FLOAT1g
float
#else
double
#endif

 quad_coef = QD[i] + QD[j] + 2 * Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			
#ifdef FLOAT1g
float
#else
double
#endif

 delta = (-G[i] - G[j]) / quad_coef;
			
#ifdef FLOAT1g
float
#else
double
#endif

 diff = alpha[i] - alpha[j];
			alpha[i] += delta;
			alpha[j] += delta;

			if (diff > 0) {
				if (alpha[j] < 0) {
					alpha[j] = 0;
					alpha[i] = diff;
				}
			} else {
				if (alpha[i] < 0) {
					alpha[i] = 0;
					alpha[j] = -diff;
				}
			}
			if (diff > C_i - C_j) {
				if (alpha[i] > C_i) {
					alpha[i] = C_i;
					alpha[j] = C_i - diff;
				}
			} else {
				if (alpha[j] > C_j) {
					alpha[j] = C_j;
					alpha[i] = C_j + diff;
				}
			}
		} else {
			
#ifdef FLOAT1g
float
#else
double
#endif

 quad_coef = QD[i] + QD[j] - 2 * Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			
#ifdef FLOAT1g
float
#else
double
#endif

 delta = (G[i] - G[j]) / quad_coef;
			
#ifdef FLOAT1g
float
#else
double
#endif

 sum = alpha[i] + alpha[j];
			alpha[i] -= delta;
			alpha[j] += delta;

			if (sum > C_i) {
				if (alpha[i] > C_i) {
					alpha[i] = C_i;
					alpha[j] = sum - C_i;
				}
			} else {
				if (alpha[j] < 0) {
					alpha[j] = 0;
					alpha[i] = sum;
				}
			}
			if (sum > C_j) {
				if (alpha[j] > C_j) {
					alpha[j] = C_j;
					alpha[i] = sum - C_j;
				}
			} else {
				if (alpha[i] < 0) {
					alpha[i] = 0;
					alpha[j] = sum;
				}
			}
		}

		// update G

		
#ifdef FLOAT1g
float
#else
double
#endif

 delta_alpha_i = alpha[i] - old_alpha_i;
		
#ifdef FLOAT1g
float
#else
double
#endif

 delta_alpha_j = alpha[j] - old_alpha_j;

		for (int k = 0; k < active_size; k++) {
			G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
		}

		// update alpha_status and G_bar

		{
			bool ui = is_upper_bound(i);
			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			update_alpha_status(j);
			int k;
			if (ui != is_upper_bound(i)) {
				Q_i = Q.get_Q(i, l);
				if (ui)
					for (k = 0; k < l; k++)
						G_bar[k] -= C_i * Q_i[k];
				else
					for (k = 0; k < l; k++)
						G_bar[k] += C_i * Q_i[k];
			}

			if (uj != is_upper_bound(j)) {
				Q_j = Q.get_Q(j, l);
				if (uj)
					for (k = 0; k < l; k++)
						G_bar[k] -= C_j * Q_j[k];
				else
					for (k = 0; k < l; k++)
						G_bar[k] += C_j * Q_j[k];
			}
		}
	}

	if (iter >= max_iter) {
		if (active_size < l) {
			// reconstruct the whole gradient to calculate objective value
			reconstruct_gradient();
			active_size = l;
			//info("*");
		}
	//	fprintf(stderr, "\nWARNING: reaching max number of iterations\n");
	}

	// calculate rho

	si->rho = calculate_rho();

	// calculate objective value
	{
		
#ifdef FLOAT1g
float
#else
double
#endif

 v = 0;
		int i;
		for (i = 0; i < l; i++)
			v += alpha[i] * (G[i] + p[i]);

		si->obj = v / 2;
	}

	// put back the solution
	{
		for (int i = 0; i < l; i++)
			alpha_[active_set[i]] = alpha[i];
	}

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	//info("\noptimization finished, #iter = %d\n", iter);

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;
}

// return 1 if already optimal, return 0 otherwise
__device__ int Solver::select_working_set(int &out_i, int &out_j) {
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	
#ifdef FLOAT1g
float
#else
double
#endif

 Gmax = -INF;
	
#ifdef FLOAT1g
float
#else
double
#endif

 Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	
#ifdef FLOAT1g
float
#else
double
#endif

 obj_diff_min = INF;

	for (int t = 0; t < active_size; t++)
		if (y[t] == +1) {
			if (!is_upper_bound(t))
				if (-G[t] >= Gmax) {
					Gmax = -G[t];
					Gmax_idx = t;
				}
		} else {
			if (!is_lower_bound(t))
				if (G[t] >= Gmax) {
					Gmax = G[t];
					Gmax_idx = t;
				}
		}

	int i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if (i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i = Q->get_Q(i, active_size);

	for (int j = 0; j < active_size; j++) {
		if (y[j] == +1) {
			if (!is_lower_bound(j)) {
				
#ifdef FLOAT1g
float
#else
double
#endif

 grad_diff = Gmax + G[j];
				if (G[j] >= Gmax2)
					Gmax2 = G[j];
				if (grad_diff > 0) {
					
#ifdef FLOAT1g
float
#else
double
#endif

 obj_diff;
					
#ifdef FLOAT1g
float
#else
double
#endif

 quad_coef = QD[i] + QD[j] - 2.0 * y[i] * Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff * grad_diff) / quad_coef;
					else
						obj_diff = -(grad_diff * grad_diff) / TAU;

					if (obj_diff <= obj_diff_min) {
						Gmin_idx = j;
						obj_diff_min = obj_diff;
					}
				}
			}
		} else {
			if (!is_upper_bound(j)) {
				
#ifdef FLOAT1g
float
#else
double
#endif

 grad_diff = Gmax - G[j];
				if (-G[j] >= Gmax2)
					Gmax2 = -G[j];
				if (grad_diff > 0) {
					
#ifdef FLOAT1g
float
#else
double
#endif

 obj_diff;
					
#ifdef FLOAT1g
float
#else
double
#endif

 quad_coef = QD[i] + QD[j] + 2.0 * y[i] * Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff * grad_diff) / quad_coef;
					else
						obj_diff = -(grad_diff * grad_diff) / TAU;

					if (obj_diff <= obj_diff_min) {
						Gmin_idx = j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if (Gmax + Gmax2 < eps)
		return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

__device__ bool Solver::be_shrunk(int i, 
#ifdef FLOAT1g
float
#else
double
#endif

 Gmax1, 
#ifdef FLOAT1g
float
#else
double
#endif

 Gmax2) {
	if (is_upper_bound(i)) {
		if (y[i] == +1)
			return (-G[i] > Gmax1);
		else
			return (-G[i] > Gmax2);
	} else if (is_lower_bound(i)) {
		if (y[i] == +1)
			return (G[i] > Gmax2);
		else
			return (G[i] > Gmax1);
	} else
		return (false);
}

__device__ 
#ifdef FLOAT1g
float
#else
double
#endif

 Solver::calculate_rho() {
	
#ifdef FLOAT1g
float
#else
double
#endif

 r;
	int nr_free = 0;
	
#ifdef FLOAT1g
float
#else
double
#endif

 ub = INF, lb = -INF, sum_free = 0;
	for (int i = 0; i < active_size; i++) {
		
#ifdef FLOAT1g
float
#else
double
#endif

 yG = y[i] * G[i];

		if (is_upper_bound(i)) {
			if (y[i] == -1)
				ub = min(ub, yG);
			else
				lb = max(lb, yG);
		} else if (is_lower_bound(i)) {
			if (y[i] == +1)
				ub = min(ub, yG);
			else
				lb = max(lb, yG);
		} else {
			++nr_free;
			sum_free += yG;
		}
	}

	if (nr_free > 0)
		r = sum_free / nr_free;
	else
		r = (ub + lb) / 2;

	return r;
}

bool gpu_t = false;
//
// Q matrices for various formulations
//
//#include "../gpu.h"

class SVR_Q: public QMatrix {
public:

	__device__ SVR_Q(const svm_problem& prob, 
#ifdef FLOAT1g
float
#else
double
#endif

 gamma)

	{
		l = prob.l;
		int l2 = l * 2;
		QD = new 
#ifdef FLOAT1g
float
#else
double
#endif

[l2];
		sign = new schar[l2];
		index = new int[l2];

		buf = new Qfloat[l2 * l2];



			x = prob.x;

			
#ifdef FLOAT1g
float
#else
double
#endif

* x_square = new 
#ifdef FLOAT1g
float
#else
double
#endif

[l];
			for (int i = 0; i < l; i++)
				x_square[i] = dot(x[i], x[i]);
			//exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j]))); -kernel_function
			QD = new 
#ifdef FLOAT1g
float
#else
double
#endif

[l2];
			for (int i = 0; i < prob.l; i++) {
				QD[i] = exp(
						-gamma
								* (x_square[i] + x_square[i]
										- 2 * dot(x[i], x[i])));

				sign[i] = 1;
				sign[i + l] = -1;
				index[i] = i;
				index[i + l] = i;
				QD[i + l] = QD[i];
			}
			data = new Qfloat[l * l];
			for (int i = 0; i < prob.l; i++)
				for (int j = 0; j < prob.l; j++) {

					data[i * l + j] = (Qfloat) (exp(
							-gamma
									* (x_square[i] + x_square[j]
											- 2 * dot(x[i], x[j]))));
				}
			delete[] x_square;

			for (int i = 0; i < l2; i++)
				for (int j = 0; j < l2; j++)
					buf[i * l + j] = (Qfloat) sign[i] * (Qfloat) sign[j]
							* data[index[i] * l + index[j]];
			delete[] data;


	}

	__device__ Qfloat *get_Q(int i, int len) const {

		return buf + i * 2 * l;
	}

	__device__ 
#ifdef FLOAT1g
float
#else
double
#endif

 dot(const svm_node &px, const svm_node &py) {
		
#ifdef FLOAT1g
float
#else
double
#endif

 sum = 0;

		int dim = px.dim;
		for (int i = 0; i < dim; i++)
			sum += px.values[i] * py.values[i];
		return sum;
	}

	__device__ 
#ifdef FLOAT1g
float
#else
double
#endif

 *get_QD() const {
		return QD;
	}

	__device__ ~SVR_Q() {

		delete[] sign;
		delete[] index;
		delete[] buf;
		delete[] QD;
	}
private:
	Qfloat *data;
	int l;
	Qfloat *buf;
	svm_node *x;
	schar *sign;
	int *index;

	
#ifdef FLOAT1g
float
#else
double
#endif

 *QD;
};

class SVC_Q: public QMatrix {
public:
	__device__ SVC_Q(const svm_problem& prob, schar *y, 
#ifdef FLOAT1g
float
#else
double
#endif

 gamma)

	{

		l = prob.l;

			x = prob.x;

			
#ifdef FLOAT1g
float
#else
double
#endif

* x_square = new 
#ifdef FLOAT1g
float
#else
double
#endif

[l];
			for (int i = 0; i < l; i++)
				x_square[i] = dot(x[i], x[i]);
//exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j]))); -kernel_function
			QD = new 
#ifdef FLOAT1g
float
#else
double
#endif

[prob.l];
			for (int i = 0; i < prob.l; i++)
				QD[i] = exp(
						-gamma
								* (x_square[i] + x_square[i]
										- 2 * dot(x[i], x[i])));
			data = new Qfloat[l * l];
			for (int i = 0; i < prob.l; i++)
				for (int j = 0; j < prob.l; j++)
					data[i * l + j] = (Qfloat) (y[i] * y[j]
							* exp(
									-gamma
											* (x_square[i] + x_square[j]
													- 2 * dot(x[i], x[j]))));
			delete[] x_square;


	}

	__device__ 
#ifdef FLOAT1g
float
#else
double
#endif

 dot(const svm_node &px, const svm_node &py) {
		
#ifdef FLOAT1g
float
#else
double
#endif

 sum = 0;

		int dim = px.dim;
		for (int i = 0; i < dim; i++)
			sum += px.values[i] * py.values[i];
		return sum;
	}

	__device__ Qfloat *get_Q(int i, int len) const {

		return data + i * l;
	}

	__device__ 
#ifdef FLOAT1g
float
#else
double
#endif

 *get_QD() const {
		return QD;
	}

	__device__ ~SVC_Q() {

		delete[] QD;
		delete[] data;
	}
private:
	int l;
	svm_node *x;
	Qfloat* data;
	
#ifdef FLOAT1g
float
#else
double
#endif

 *QD;
};

//
// construct and solve various formulations
//
__device__ static void solve_c_svc(const svm_problem *prob,
		const svm_parameter* param, 
#ifdef FLOAT1g
float
#else
double
#endif

 *alpha, Solver::SolutionInfo* si,
		
#ifdef FLOAT1g
float
#else
double
#endif

 Cp, 
#ifdef FLOAT1g
float
#else
double
#endif

 Cn) {
	int l = prob->l;
	
#ifdef FLOAT1g
float
#else
double
#endif

 *minus_ones = new 
#ifdef FLOAT1g
float
#else
double
#endif

[l];
	schar *y = new schar[l];

	int i;

	for (i = 0; i < l; i++) {
		alpha[i] = 0;
		minus_ones[i] = -1;
		if (prob->y[i] > 0)
			y[i] = +1;
		else
			y[i] = -1;
	}

	Solver s;
	//TODO
	SVC_Q svc = SVC_Q(*prob, y, param->gamma);
	s.Solve(l, svc, minus_ones, y, alpha, Cp, Cn, param->eps, si, 0);

	
#ifdef FLOAT1g
float
#else
double
#endif

 sum_alpha = 0;
	for (i = 0; i < l; i++)
		sum_alpha += alpha[i];

	if (Cp == Cn)
		//info("nu = %f\n", sum_alpha / (Cp * prob->l));

	for (i = 0; i < l; i++)
		alpha[i] *= y[i];

	delete[] minus_ones;
	delete[] y;
}

//
// decision_function
//
struct decision_function {
	
#ifdef FLOAT1g
float
#else
double
#endif

 *alpha;
	
#ifdef FLOAT1g
float
#else
double
#endif

 rho;
};

__device__ static void solve_epsilon_svr(const svm_problem *prob,
		const svm_parameter *param, 
#ifdef FLOAT1g
float
#else
double
#endif

 *alpha, Solver::SolutionInfo* si) {
	int l = prob->l;
	
#ifdef FLOAT1g
float
#else
double
#endif

 *alpha2 = new 
#ifdef FLOAT1g
float
#else
double
#endif

[2 * l];
	
#ifdef FLOAT1g
float
#else
double
#endif

 *linear_term = new 
#ifdef FLOAT1g
float
#else
double
#endif

[2 * l];
	schar *y = new schar[2 * l];
	int i;

	for (i = 0; i < l; i++) {
		alpha2[i] = 0;
		linear_term[i] = param->p - prob->y[i];
		y[i] = 1;

		alpha2[i + l] = 0;
		linear_term[i + l] = param->p + prob->y[i];
		y[i + l] = -1;
	}

	Solver s;
	SVR_Q svc = SVR_Q(*prob, param->gamma);
	s.Solve(2 * l, svc, linear_term, y, alpha2, param->C, param->C, param->eps,
			si, param->shrinking);

	
#ifdef FLOAT1g
float
#else
double
#endif

 sum_alpha = 0;
	for (i = 0; i < l; i++) {
		alpha[i] = alpha2[i] - alpha2[i + l];
		sum_alpha += fabs(alpha[i]);
	}
	//info("nu = %f\n", sum_alpha / (param->C * l));

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

__device__  static decision_function svm_train_one(const svm_problem *prob,
		const svm_parameter *param, 
#ifdef FLOAT1g
float
#else
double
#endif

 Cp, 
#ifdef FLOAT1g
float
#else
double
#endif

 Cn) {
	
#ifdef FLOAT1g
float
#else
double
#endif

 *alpha = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

,prob->l);
	Solver::SolutionInfo si;

	switch (param->svm_type) {
	case C_SVC:
		solve_c_svc(prob, param, alpha, &si, Cp, Cn);
		break;

	case EPSILON_SVR:
		solve_epsilon_svr(prob, param, alpha, &si);
		break;

	}

	//info("obj = %f, rho = %f\n", si.obj, si.rho);

	// output SVs

	int nSV = 0;
	int nBSV = 0;
	for (int i = 0; i < prob->l; i++) {
		if (fabs(alpha[i]) > 0) {
			++nSV;
			if (prob->y[i] > 0) {
				if (fabs(alpha[i]) >= si.upper_bound_p)
					++nBSV;
			} else {
				if (fabs(alpha[i]) >= si.upper_bound_n)
					++nBSV;
			}
		}
	}

	//info("nSV = %d, nBSV = %d\n", nSV, nBSV);

	decision_function f;
	f.alpha = alpha;
	f.rho = si.rho;
	return f;
}


// Return parameter of a Laplace distribution
// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
__device__ static void svm_group_classes(const svm_problem *prob,
		int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret,
		int *perm) {
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for (i = 0; i < l; i++) {
		int this_label = (int) prob->y[i];
		int j;
		for (j = 0; j < nr_class; j++) {
			if (this_label == label[j]) {
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if (j == nr_class) {
			if (nr_class == max_nr_class) {
							max_nr_class *= 2;
							int* new_label = new int[max_nr_class];
							int* new_count = new int[max_nr_class];
							for (int i = 0; i < nr_class; i++) {
								new_label[i] = label[i];
								new_count[i] = count[i];
							}
							free(label);
							free(count);
							label = new_label;
							count = new_count;
						}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set.
	// However, for two-class sets with -1/+1 labels and -1 appears first,
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1) {
		swap(label[0], label[1]);
		swap(count[0], count[1]);
		for (i = 0; i < l; i++) {
			if (data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for (i = 1; i < nr_class; i++)
		start[i] = start[i - 1] + count[i - 1];
	for (i = 0; i < l; i++) {
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for (i = 1; i < nr_class; i++)
		start[i] = start[i - 1] + count[i - 1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

//
// Interface functions
//
__device__ svm_model *svm_train(const svm_problem *prob,
		const svm_parameter *param) {


	svm_model *model = Malloc(svm_model,1);
	model->param = *param;
	model->free_sv = 0;	// XXX

	if (param->svm_type == ONE_CLASS || param->svm_type == EPSILON_SVR
			|| param->svm_type == NU_SVR) {
		// regression or one-class-svm
		model->nr_class = 2;
		model->label = NULL;
		model->nSV = NULL;
		model->probA = NULL;
		model->probB = NULL;
		model->sv_coef = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

 *,1);

		decision_function f = svm_train_one(prob, param, 0, 0);
		model->rho = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

,1);
		model->rho[0] = f.rho;

		int nSV = 0;
		int i;
		for (i = 0; i < prob->l; i++)
			if (fabs(f.alpha[i]) > 0)
				++nSV;
		model->l = nSV;
#ifdef _DENSE_REP
		model->SV = Malloc(svm_node,nSV);
#else
		model->SV = Malloc(svm_node *,nSV);
#endif
		model->sv_coef[0] = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

,nSV);
		model->sv_indices = Malloc(int,nSV);
		int j = 0;
		for (i = 0; i < prob->l; i++)
			if (fabs(f.alpha[i]) > 0) {
				model->SV[j] = prob->x[i];
				model->sv_coef[0][j] = f.alpha[i];
				model->sv_indices[j] = i + 1;
				++j;
			}

		free(f.alpha);
	} else {

		//param->shrinking=0;
		// classification
		int l = prob->l;
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		svm_group_classes(prob, &nr_class, &label, &start, &count, perm);
	//	if (nr_class == 1)
			//info(
			//		"WARNING: training data in only one class. See README for details.\n");

		svm_node *x = Malloc(svm_node,l);

		int i;
		for (i = 0; i < l; i++)
			x[i] = prob->x[perm[i]];

		// calculate weighted C

		
#ifdef FLOAT1g
float
#else
double
#endif

 *weighted_C = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

, nr_class);
		for (i = 0; i < nr_class; i++)
			weighted_C[i] = param->C;
		for (i = 0; i < param->nr_weight; i++) {
			int j;
			for (j = 0; j < nr_class; j++)
				if (param->weight_label[i] == label[j])
					break;
			if (j != nr_class)
						weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models

		bool *nonzero = Malloc(bool,l);
		for (i = 0; i < l; i++)
			nonzero[i] = false;
		decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

		
#ifdef FLOAT1g
float
#else
double
#endif

 *probA = NULL, *probB = NULL;
		if (param->probability) {
			probA = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

,nr_class*(nr_class-1)/2);
			probB = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

,nr_class*(nr_class-1)/2);
		}

		int p = 0;
		for (i = 0; i < nr_class; i++)
			for (int j = i + 1; j < nr_class; j++) {
				svm_problem sub_prob;
				int si = start[i], sj = start[j];
				int ci = count[i], cj = count[j];
				sub_prob.l = ci + cj;
#ifdef _DENSE_REP
				sub_prob.x = Malloc(svm_node,sub_prob.l);
#else
				sub_prob.x = Malloc(svm_node *,sub_prob.l);
#endif
				sub_prob.y = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

,sub_prob.l);
				int k;
				for (k = 0; k < ci; k++) {
					sub_prob.x[k] = x[si + k];
					sub_prob.y[k] = +1;
				}
				for (k = 0; k < cj; k++) {
					sub_prob.x[ci + k] = x[sj + k];
					sub_prob.y[ci + k] = -1;
				}

				f[p] = svm_train_one(&sub_prob, param, weighted_C[i],
						weighted_C[j]);
				for (k = 0; k < ci; k++)
					if (!nonzero[si + k] && fabs(f[p].alpha[k]) > 0)
						nonzero[si + k] = true;
				for (k = 0; k < cj; k++)
					if (!nonzero[sj + k] && fabs(f[p].alpha[ci + k]) > 0)
						nonzero[sj + k] = true;
				free(sub_prob.x);
				free(sub_prob.y);
				++p;
			}

		// build output

		model->nr_class = nr_class;

		model->label = Malloc(int,nr_class);
		for (i = 0; i < nr_class; i++)
			model->label[i] = label[i];

		model->rho = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

,nr_class*(nr_class-1)/2);
		for (i = 0; i < nr_class * (nr_class - 1) / 2; i++)
			model->rho[i] = f[i].rho;

		if (param->probability) {
			model->probA = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

,nr_class*(nr_class-1)/2);
			model->probB = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

,nr_class*(nr_class-1)/2);
			for (i = 0; i < nr_class * (nr_class - 1) / 2; i++) {
				model->probA[i] = probA[i];
				model->probB[i] = probB[i];
			}
		} else {
			model->probA = NULL;
			model->probB = NULL;
		}

		int total_sv = 0;
		int *nz_count = Malloc(int,nr_class);
		model->nSV = Malloc(int,nr_class);
		for (i = 0; i < nr_class; i++) {
			int nSV = 0;
			for (int j = 0; j < count[i]; j++)
				if (nonzero[start[i] + j]) {
					++nSV;
					++total_sv;
				}
			model->nSV[i] = nSV;
			nz_count[i] = nSV;
		}

		//info("Total nSV = %d\n", total_sv);

		model->l = total_sv;
#ifdef _DENSE_REP
		model->SV = Malloc(svm_node,total_sv);
#else
		model->SV = Malloc(svm_node *,total_sv);
#endif
		model->sv_indices = Malloc(int,total_sv);
		p = 0;
		for (i = 0; i < l; i++)
			if (nonzero[i]) {
				model->SV[p] = x[i];
				model->sv_indices[p++] = perm[i] + 1;
			}

		int *nz_start = Malloc(int,nr_class);
		nz_start[0] = 0;
		for (i = 1; i < nr_class; i++)
			nz_start[i] = nz_start[i - 1] + nz_count[i - 1];

		model->sv_coef = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

 *,nr_class-1);
		for (i = 0; i < nr_class - 1; i++)
			model->sv_coef[i] = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

,total_sv);

		p = 0;
		for (i = 0; i < nr_class; i++)
			for (int j = i + 1; j < nr_class; j++) {
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = count[i];
				int cj = count[j];

				int q = nz_start[i];
				int k;
				for (k = 0; k < ci; k++)
					if (nonzero[si + k])
						model->sv_coef[j - 1][q++] = f[p].alpha[k];
				q = nz_start[j];
				for (k = 0; k < cj; k++)
					if (nonzero[sj + k])
						model->sv_coef[i][q++] = f[p].alpha[ci + k];
				++p;
			}

		free(label);
		free(probA);
		free(probB);
		free(count);
		free(perm);
		free(start);
		free(x);
		free(weighted_C);
		free(nonzero);
		for (i = 0; i < nr_class * (nr_class - 1) / 2; i++)
			free(f[i].alpha);
		free(f);
		free(nz_count);
		free(nz_start);

	}
	return model;
}


__device__ 
#ifdef FLOAT1g
float
#else
double
#endif

 svm_predict_values(const svm_model *model, const svm_node *x,
		
#ifdef FLOAT1g
float
#else
double
#endif

* dec_values) {
	int i;

	if (model->param.svm_type == ONE_CLASS
			|| model->param.svm_type == EPSILON_SVR
			|| model->param.svm_type == NU_SVR) {
		
#ifdef FLOAT1g
float
#else
double
#endif

 *sv_coef = model->sv_coef[0];
		
#ifdef FLOAT1g
float
#else
double
#endif

 sum = 0;

		for (i = 0; i < model->l; i++)

			sum += sv_coef[i] * k_function(x, model->SV + i, model->param);

		sum -= model->rho[0];
		*dec_values = sum;

		return sum;
	} else {

		int nr_class = model->nr_class;
		int l = model->l;

		
#ifdef FLOAT1g
float
#else
double
#endif

 *kvalue = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

,l);
		for (i = 0; i < l; i++)

			kvalue[i] = k_function(x, model->SV + i, model->param);

		int *start = Malloc(int,nr_class);
		start[0] = 0;
		for (i = 1; i < nr_class; i++)
			start[i] = start[i - 1] + model->nSV[i - 1];

		int *vote = Malloc(int,nr_class);
		for (i = 0; i < nr_class; i++)
			vote[i] = 0;

		int p = 0;
		for (i = 0; i < nr_class; i++)
			for (int j = i + 1; j < nr_class; j++) {
				
#ifdef FLOAT1g
float
#else
double
#endif

 sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model->nSV[i];
				int cj = model->nSV[j];

				int k;
				
#ifdef FLOAT1g
float
#else
double
#endif

 *coef1 = model->sv_coef[j - 1];
				
#ifdef FLOAT1g
float
#else
double
#endif

 *coef2 = model->sv_coef[i];
				for (k = 0; k < ci; k++)
					sum += coef1[si + k] * kvalue[si + k];
				for (k = 0; k < cj; k++)
					sum += coef2[sj + k] * kvalue[sj + k];
				sum -= model->rho[p];
				dec_values[p] = sum;

				if (dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				p++;
			}

		int vote_max_idx = 0;
		for (i = 1; i < nr_class; i++)
			if (vote[i] > vote[vote_max_idx])
				vote_max_idx = i;

		free(kvalue);
		free(start);
		free(vote);
		return model->label[vote_max_idx];
	}
}

__device__ 
#ifdef FLOAT1g
float
#else
double
#endif

 svm_predict(const svm_model *model, const svm_node *x) {
	int nr_class = model->nr_class;
	
#ifdef FLOAT1g
float
#else
double
#endif

 *dec_values;
	if (model->param.svm_type == ONE_CLASS
			|| model->param.svm_type == EPSILON_SVR
			|| model->param.svm_type == NU_SVR)
		dec_values = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

, 1);
	else
		dec_values = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

, nr_class*(nr_class-1)/2);
	
#ifdef FLOAT1g
float
#else
double
#endif

 pred_result = svm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}


#define max(x,y) (((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))

__device__ 
#ifdef FLOAT1g
float
#else
double
#endif

* scale_gpu(int row, int col, float* array, 
#ifdef FLOAT1g
float
#else
double
#endif

 max_scale,
		
#ifdef FLOAT1g
float
#else
double
#endif

 min_scale) {
	
#ifdef FLOAT1g
float
#else
double
#endif

* ret_row = new 
#ifdef FLOAT1g
float
#else
double
#endif

[col];
	
#ifdef FLOAT1g
float
#else
double
#endif

 minV = 0;
	
#ifdef FLOAT1g
float
#else
double
#endif

 maxV = 0;
	for (int c = 1; c < col; ++c) {
		minV = min(minV, array[row * col + c]);
		maxV = max(maxV,(
#ifdef FLOAT1g
float
#else
double
#endif

) array[row * col + c]);
	}
	for (int c = 0; c < col; ++c) {
		
#ifdef FLOAT1g
float
#else
double
#endif

 value = (
#ifdef FLOAT1g
float
#else
double
#endif

) array[row * col + c];
		if (value == minV)
			value = min_scale;
		else if (value == maxV)
			value = max_scale;
		else
			value = min_scale
					+ (max_scale - min_scale) * (value - minV) / (maxV - minV);
		ret_row[c] = value;
	}

	return ret_row;
}

const char *sSDKsample = "newdelete";

struct svm_p {
	float C;
	float gamma;
};

__global__ void kernel(float* gpu0, int size0, 
#ifdef FLOAT1g
float
#else
double
#endif

* gpu1, int size1,
		float* testData, int test_size, 
#ifdef FLOAT1g
float
#else
double
#endif

* gpu_out, svm_p* array_param,
		int dim,int param_length,bool classification) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
// you should include check for elements
	if (idx<param_length) {
	svm_parameter param;
	svm_problem prob;

// if your block*thraad_in_block multiplication > your array size
if (classification)
	param.svm_type = C_SVC;
		else
			param.svm_type = EPSILON_SVR; //C_SVC;

	param.kernel_type = RBF;
	param.degree = 3;
	//	param.gamma = currentGTamma;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 1;
	//	param.C = currentC;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 0;
	param.probability = 0;

	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.C = array_param[idx].C;
	param.gamma = array_param[idx].gamma;
//(N your array elements count or your parallel computations) - if (idx<N) {//your code}
	//svm_problem prob;

	prob.l = size1;
	prob.y = gpu1;
	prob.x = new svm_node[size1];
	//int dim = size0 / size1;
	for (int i = 0; i < size1; i++) {
		prob.x[i].dim = dim;
		prob.x[i].values = new 
#ifdef FLOAT1g
float
#else
double
#endif

[dim];


		for (int j = 0; j < dim; j++)
			prob.x[i].values[j] = (
#ifdef FLOAT1g
float
#else
double
#endif

) gpu0[i * dim + j];

	}

	svm_model* model = svm_train(&prob, &param);

	
#ifdef FLOAT1g
float
#else
double
#endif

* dec_values = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

,test_size );

	//--	Test Data
	//		svm_node* svm_t = new svm_node[test_size + 1];

	for (size_t i = 0; i < test_size; i++) {
		svm_node* svm_t = new svm_node;

		svm_t->dim = dim;
		svm_t->values = Malloc(
#ifdef FLOAT1g
float
#else
double
#endif

,dim);
		for (int c = 0; c < dim; ++c) {

			svm_t->values[c] = (
#ifdef FLOAT1g
float
#else
double
#endif

) testData[i * dim + c];//---CHECK HERE-debug here
		}

		dec_values[i] = svm_predict(model, svm_t);


		delete[] svm_t->values;
		delete svm_t;
	}

	memcpy(gpu_out + (test_size * idx), dec_values,
			sizeof(
#ifdef FLOAT1g
float
#else
double
#endif

) * (test_size));
	//printf("%d", model->l);
	//gpu_out= dec_values;
	}
}

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

/*if you want to separate kernel*/ //#include <kernel.cu>
//uncomment if you want to make function external
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
*trainClass, float* train, int dim,
		float* svm_train, int len, svm_p* array_param, int param_length,bool isClassification) {
	int block = 0;
	int thread = 0;

	int cuda_device = 0;

	printf("%s Starting...\n\n", sSDKsample);

	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDevice(&cuda_device));
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

	if (deviceProp.major < 2) {
		printf(
				"> This GPU with Compute Capability %d.%d does not meet minimum requirements.\n",
				deviceProp.major, deviceProp.minor);
		printf(
				"> A GPU with Compute Capability >= 2.0 is required to run %s.\n",
				sSDKsample);
		printf("> Test will not run.  Exiting.\n");
		exit(EXIT_SUCCESS);
	}

	// set the heap size for device size new/delete to 128 MB
#if CUDART_VERSION >= 4000
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * (1 << 20));
#else
	cudaThreadSetLimit(cudaLimitMallocHeapSize, 1024 * (1 << 20));
#endif

	//cudaDeviceProp deviceProp;
		//cudaGetDeviceProperties(&deviceProp, 0);
		int max_threads = deviceProp.maxThreadsPerBlock;
	//	int d_blocks = deviceProp.multiProcessorCount;
		//int threads_m = deviceProp.

	float*gpu0;
	float* gpu_testData;
	
#ifdef FLOAT1g
float
#else
double
#endif

* y;
	svm_p* svm_para;
	cudaMalloc(&svm_para, sizeof(svm_p) * param_length);

	block = param_length/100+1;
	thread = 100;

	
#ifdef FLOAT1g
float
#else
double
#endif

*gpu_out;
	cudaMalloc(&y, l * sizeof(
#ifdef FLOAT1g
float
#else
double
#endif

));
	cudaMalloc(&gpu_out, len * sizeof(
#ifdef FLOAT1g
float
#else
double
#endif

) * param_length);
	cudaMalloc(&gpu0, l * dim * sizeof(float));

	cudaMalloc(&gpu_testData, len * dim * sizeof(float));
	cudaDeviceSetCacheConfig (cudaFuncCachePreferL1);
	checkCudaErrors(
			cudaMemcpy(svm_para, array_param, param_length * sizeof(svm_p),
					cudaMemcpyHostToDevice));

	checkCudaErrors(
			cudaMemcpy(y, trainClass, l * sizeof(
#ifdef FLOAT1g
float
#else
double
#endif

),
					cudaMemcpyHostToDevice));

	checkCudaErrors(
			cudaMemcpy(gpu0, train, l * dim * sizeof(float),
					cudaMemcpyHostToDevice));

	checkCudaErrors(
			cudaMemcpy(gpu_testData, svm_train, len * dim * sizeof(float),
					cudaMemcpyHostToDevice));

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	//	cudaStream_t *streams = (cudaStream_t *) malloc(param_length * sizeof(cudaStream_t));
	//param_length=2;
	//for (int i = 0; i < param_length; i++)
	//{
	//  checkCudaErrors(cudaStreamCreate(&(streams[i])));
	//}

	//  for (int i = 0; i < param_length; i++)
	//   {
	//checkCudaErrors(cudaStreamCreate(&(streams[i])));

	kernel<<<block, thread>>>(gpu0,l * dim,y,l,gpu_testData,len,gpu_out,svm_para,dim,param_length,isClassification);
	//}
	printf("Working...\n");
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());

	
#ifdef FLOAT1g
float
#else
double
#endif

* result = new 
#ifdef FLOAT1g
float
#else
double
#endif

[len * param_length];
	checkCudaErrors(
			cudaMemcpy(result, gpu_out, sizeof(
#ifdef FLOAT1g
float
#else
double
#endif

) * len * param_length,
					cudaMemcpyDeviceToHost));

	//	for (int i=0;i<len;i++)
	//	printf("predicted %d class: %f \n",i,result[i]);

	checkCudaErrors(cudaFree(y));
	checkCudaErrors(cudaFree(gpu0));
	checkCudaErrors(cudaFree(gpu_testData));
	checkCudaErrors(cudaFree(gpu_out));
	checkCudaErrors(cudaFree(svm_para));
//	cudaDeviceReset();
	/*for (int i=0;i<block*thread;i++) {
	 for (int j=0;j<len;j++)
	 {printf("%d ",(int)result[j*i+i]);

	 }printf("\n");}*/
	printf("Done...\n");
	CUDA_CHECK_RETURN(cudaDeviceReset());
	return result;
//	printf("Created");

}
