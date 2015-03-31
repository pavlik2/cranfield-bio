/*
 * svm_function2.cpp
 *
 *  Created on: 16 Jul 2013
 *      Author: pavel
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
//#include <stdarg.h>
//#include <limits.h>
//#include <locale.h>
#  define INT_MIN	(-INT_MAX - 1)
#  define INT_MAX	2147483647

#include "svm_function.h"

int count=0;
void passed() {
	printf("passed: %d",count++);
}

typedef float Qfloat;
typedef signed char schar;
#ifndef min
template<class T> static inline T min(T x, T y) {
	return (x < y) ? x : y;
}
#endif
#ifndef max
template<class T> static inline T max(T x, T y) {
	return (x > y) ? x : y;
}
#endif
template<class T> static inline void swap(T& x, T& y) {
	T t = x;
	x = y;
	y = t;
}
template<class S, class T> static inline void clone(T*& dst, S* src, int n) {
	dst = new T[n];
	memcpy((void *) dst, (void *) src, sizeof(T) * n);
}

#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct decision_function {
	double *alpha;
	double rho;
};



svm_node* x;
double* x_square;

schar* sign;
int* indexx;

Qfloat* get_Q(int i, int len);
void swap_indexx(int i, int j);
double kernel_function(int i, int j);

void init_kernel(const svm_problem& prob, const svm_parameter& param);


void svm_free_model_content(svm_model* model_ptr) {
	if (model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)

		for (int i = 0; i < model_ptr->l; i++)
			free(model_ptr->SV[i].values);

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

void svm_free_and_destroy_model(svm_model** model_ptr_ptr) {
	if (model_ptr_ptr != NULL && *model_ptr_ptr != NULL) {
		svm_free_model_content(*model_ptr_ptr);
		free(*model_ptr_ptr);
		*model_ptr_ptr = NULL;
	}
}

void svm_destroy_param(svm_parameter* param) {
	free(param->weight_label);
	free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob,
		const svm_parameter *param) {
	// svm_type

	int svm_type = param->svm_type;
	if (svm_type != C_SVC && svm_type != NU_SVC && svm_type != ONE_CLASS
			&& svm_type != EPSILON_SVR && svm_type != NU_SVR)
		return "unknown svm type";

	// kernel_type, degree

	int kernel_type = param->kernel_type;
	if (kernel_type != LINEAR && kernel_type != POLY && kernel_type != RBF
			&& kernel_type != SIGMOID && kernel_type != PRECOMPUTED)
		return "unknown kernel type";

	if (param->gamma < 0)
		return "gamma < 0";

	if (param->degree < 0)
		return "degree of polynomial kernel < 0";

	// cache_size,eps,C,nu,p,shrinking

	if (param->cache_size <= 0)
		return "cache_size <= 0";

	if (param->eps <= 0)
		return "eps <= 0";

	if (svm_type == C_SVC || svm_type == EPSILON_SVR || svm_type == NU_SVR)
		if (param->C <= 0)
			return "C <= 0";

	if (svm_type == NU_SVC || svm_type == ONE_CLASS || svm_type == NU_SVR)
		if (param->nu <= 0 || param->nu > 1)
			return "nu <= 0 or nu > 1";

	if (svm_type == EPSILON_SVR)
		if (param->p < 0)
			return "p < 0";

	if (param->shrinking != 0 && param->shrinking != 1)
		return "shrinking != 0 and shrinking != 1";

	if (param->probability != 0 && param->probability != 1)
		return "probability != 0 and probability != 1";

	if (param->probability == 1 && svm_type == ONE_CLASS)
		return "one-class SVM probability output not supported yet";

	// check whether nu-svc is feasible

	if (svm_type == NU_SVC) {
		int l = prob->l;
		int max_nr_class = 16;
		int nr_class = 0;
		int *label = Malloc(int,max_nr_class);
		int *count = Malloc(int,max_nr_class);

		int i;
		for (i = 0; i < l; i++) {
			int this_label = (int) prob->y[i];
			int j;
			for (j = 0; j < nr_class; j++)
				if (this_label == label[j]) {
					++count[j];
					break;
				}
			if (j == nr_class) {
				if (nr_class == max_nr_class) {
					max_nr_class *= 2;
					label = (int *) realloc(label, max_nr_class * sizeof(int));
					count = (int *) realloc(count, max_nr_class * sizeof(int));
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}

		for (i = 0; i < nr_class; i++) {
			int n1 = count[i];
			for (int j = i + 1; j < nr_class; j++) {
				int n2 = count[j];
				if (param->nu * (n1 + n2) / 2 > min(n1, n2)) {
					free(label);
					free(count);
					return "specified nu is infeasible";
				}
			}
		}
		free(label);
		free(count);
	}

	return NULL;
}

//TODO variables
int active_size;
schar *y;
double *G;		// gradient of objective function
enum {
	LOWER_BOUND, UPPER_BOUND, FREE
};


char* alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
double *alpha;
//const QMatrix *Q;
double *QD;
double eps;
double Cp, Cn;
double *p;
int *active_set;
double *G_bar;		// gradient, if we treat free variables as 0
int l;
int ll;
bool unshrink;	// XXX

bool is_free(int i) {
	return alpha_status[i] == FREE;
}
void swap_indexx(int i, int j) {
//		Q->swap_indexx(i,j);
	swap(y[i], y[j]);
	swap(G[i], G[j]);
	swap(alpha_status[i], alpha_status[j]);
	swap(alpha[i], alpha[j]);
	swap(p[i], p[j]);
	swap(active_set[i], active_set[j]);
	swap(G_bar[i], G_bar[j]);
	swap(x[i],x[j]);
			if(x_square) swap(x_square[i],x_square[j]);
			swap(sign[i],sign[j]);
					swap(indexx[i],indexx[j]);
					swap(QD[i],QD[j]);
}

double get_C(int i) {
	return (y[i] > 0) ? Cp : Cn;
}
void update_alpha_status(int i) {
	if (alpha[i] >= get_C(i))
		alpha_status[i] = UPPER_BOUND;
	else if (alpha[i] <= 0)
		alpha_status[i] = LOWER_BOUND;
	else
		alpha_status[i] = FREE;
}
bool is_upper_bound(int i) {
	return alpha_status[i] == UPPER_BOUND;
}
bool is_lower_bound(int i) {
	return alpha_status[i] == LOWER_BOUND;
}

bool be_shrunk(int i, double Gmax1, double Gmax2) {
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

// return 1 if already optimal, return 0 otherwise
int select_working_set(int &out_i, int &out_j) {
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmax = -INF;
	double Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

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
		Q_i = get_Q(i, active_size);

	for (int j = 0; j < active_size; j++) {
		if (y[j] == +1) {
			if (!is_lower_bound(j)) {
				double grad_diff = Gmax + G[j];
				if (G[j] >= Gmax2)
					Gmax2 = G[j];
				if (grad_diff > 0) {
					double obj_diff;
					double quad_coef = QD[i] + QD[j] - 2.0 * y[i] * Q_i[j];
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
				double grad_diff = Gmax - G[j];
				if (-G[j] >= Gmax2)
					Gmax2 = -G[j];
				if (grad_diff > 0) {
					double obj_diff;
					double quad_coef = QD[i] + QD[j] + 2.0 * y[i] * Q_i[j];
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
delete[] Q_i;
	if (Gmax + Gmax2 < eps)
		return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

void reconstruct_gradient() {
	// reconstruct inactive elements of G from G_bar and free variables

	if (active_size == ll)
		return;

	int i, j;
	int nr_free = 0;

	for (j = active_size; j < ll; j++)
		G[j] = G_bar[j] + p[j];

	for (j = 0; j < active_size; j++)
		if (is_free(j))
			nr_free++;

	if (nr_free * ll > 2 * active_size * (ll - active_size)) {
		for (i = active_size; i < ll; i++) {
			const Qfloat *Q_i = get_Q(i, active_size);
			for (j = 0; j < active_size; j++)
				if (is_free(j))
					G[i] += alpha[j] * Q_i[j];
			delete[] Q_i;
		}
	} else {
		for (i = 0; i < active_size; i++)
			if (is_free(i)) {
				const Qfloat *Q_i = get_Q(i, ll);
				double alpha_i = alpha[i];
				for (j = active_size; j < ll; j++)
					G[j] += alpha_i * Q_i[j];
				delete[] Q_i;
			}
	}
}

//double dot(const svm_node &px, const svm_node &py)
//{
//	double sum = 0;
//
//	int dim = min(px.dim, py.dim);
//	for (int i = 0; i < dim; i++)
//		sum += px.values[i] * py.values[i];
//	return sum;
//}

double dot(svm_node px, svm_node py) {
	double sum = 0;

	int dim = min(px.dim, py.dim);
	for (int i = 0; i < dim; i++)
		sum += (px.values)[i] * (py.values)[i];
	return sum;
}

double gamma1;

Qfloat* get_Q(int i, int len) {

	Qfloat *data = new Qfloat[l];
	int j, real_i = indexx[i];


		for (j = 0; j < l; j++)
			data[j] = (Qfloat) kernel_function(real_i,j);


	// reorder and copy
	Qfloat *buf = new Qfloat[len];
	//next_buffer = 1 - next_buffer;
	schar si = sign[i];
	for (j = 0; j < len; j++)
		buf[j] = (Qfloat) si * (Qfloat) sign[j] * data[indexx[j]];
	delete[] data;
	return buf;

}
struct SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
	};


void do_shrinking()
{
	int i;
	double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(i=0;i<active_size;i++)
	{
		if(y[i]==+1)
		{
			if(!is_upper_bound(i))
			{
				if(-G[i] >= Gmax1)
					Gmax1 = -G[i];
			}
			if(!is_lower_bound(i))
			{
				if(G[i] >= Gmax2)
					Gmax2 = G[i];
			}
		}
		else
		{
			if(!is_upper_bound(i))
			{
				if(-G[i] >= Gmax2)
					Gmax2 = -G[i];
			}
			if(!is_lower_bound(i))
			{
				if(G[i] >= Gmax1)
					Gmax1 = G[i];
			}
		}
	}

	if(unshrink == false && Gmax1 + Gmax2 <= eps*10)
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
		////info("*");
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2))
				{
					swap_indexx(i,active_size);
					break;
				}
				active_size--;
			}
		}
}


double calculate_rho()
{
	double r;
	int nr_free = 0;
	double ub = INF, lb = -INF, sum_free = 0;
	for(int i=0;i<active_size;i++)
	{
		double yG = y[i]*G[i];

		if(is_upper_bound(i))
		{
			if(y[i]==-1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else
		{
			++nr_free;
			sum_free += yG;
		}
	}

	if(nr_free>0)
		r = sum_free/nr_free;
	else
		r = (ub+lb)/2;

	return r;
}


void Solve(int l,  double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking) {


	{
			alpha_status = new char[l];
			for(int i=0;i<l;i++)
				update_alpha_status(i);
		}

		// initialize active set (for shrinking)
		{
			active_set = new int[l];
			for(int i=0;i<l;i++)
				active_set[i] = i;
			active_size = l;
		}

		// initialize gradient
		{
			G = new double[l];
			G_bar = new double[l];
			int i;
			for(i=0;i<l;i++)
			{
				G[i] = p[i];
				G_bar[i] = 0;
			}
			for(i=0;i<l;i++)
				if(!is_lower_bound(i))
				{
					const Qfloat *Q_i = get_Q(i,l);
					double alpha_i = alpha[i];
					int j;
					for(j=0;j<l;j++)
						G[j] += alpha_i*Q_i[j];
					if(is_upper_bound(i))
						for(j=0;j<l;j++)
							G_bar[j] += get_C(i) * Q_i[j];
					delete[] Q_i;
				}
		}

		// optimization step

			int iter = 0;
			int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
			int counter = min(l,1000)+1;

			while(iter < max_iter)
			{
				// show progress and do shrinking

				if(--counter == 0)
				{
					counter = min(l,1000);
					if(shrinking) do_shrinking();
					//info(".");
				}

				int i,j;
				if(select_working_set(i,j)!=0)
				{
					// reconstruct the whole gradient
					reconstruct_gradient();
					// reset active set size and check
					active_size = l;
					//info("*");
					if(select_working_set(i,j)!=0)
						break;
					else
						counter = 1;	// do shrinking next iteration
				}

				++iter;

				// update alpha[i] and alpha[j], handle bounds carefully

				const Qfloat *Q_i = get_Q(i,active_size);
				const Qfloat *Q_j = get_Q(j,active_size);

				double C_i = get_C(i);
				double C_j = get_C(j);

				double old_alpha_i = alpha[i];
				double old_alpha_j = alpha[j];

				if(y[i]!=y[j])
				{
					double quad_coef = QD[i]+QD[j]+2*Q_i[j];
					if (quad_coef <= 0)
						quad_coef = TAU;
					double delta = (-G[i]-G[j])/quad_coef;
					double diff = alpha[i] - alpha[j];
					alpha[i] += delta;
					alpha[j] += delta;

					if(diff > 0)
					{
						if(alpha[j] < 0)
						{
							alpha[j] = 0;
							alpha[i] = diff;
						}
					}
					else
					{
						if(alpha[i] < 0)
						{
							alpha[i] = 0;
							alpha[j] = -diff;
						}
					}
					if(diff > C_i - C_j)
					{
						if(alpha[i] > C_i)
						{
							alpha[i] = C_i;
							alpha[j] = C_i - diff;
						}
					}
					else
					{
						if(alpha[j] > C_j)
						{
							alpha[j] = C_j;
							alpha[i] = C_j + diff;
						}
					}
				}
				else
				{
					double quad_coef = QD[i]+QD[j]-2*Q_i[j];
					if (quad_coef <= 0)
						quad_coef = TAU;
					double delta = (G[i]-G[j])/quad_coef;
					double sum = alpha[i] + alpha[j];
					alpha[i] -= delta;
					alpha[j] += delta;

					if(sum > C_i)
					{
						if(alpha[i] > C_i)
						{
							alpha[i] = C_i;
							alpha[j] = sum - C_i;
						}
					}
					else
					{
						if(alpha[j] < 0)
						{
							alpha[j] = 0;
							alpha[i] = sum;
						}
					}
					if(sum > C_j)
					{
						if(alpha[j] > C_j)
						{
							alpha[j] = C_j;
							alpha[i] = sum - C_j;
						}
					}
					else
					{
						if(alpha[i] < 0)
						{
							alpha[i] = 0;
							alpha[j] = sum;
						}
					}
				}

				// update G

				double delta_alpha_i = alpha[i] - old_alpha_i;
				double delta_alpha_j = alpha[j] - old_alpha_j;

				for(int k=0;k<active_size;k++)
				{
					G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
				}
				delete[] Q_i;
				delete[] Q_j;
				// update alpha_status and G_bar

				{
					bool ui = is_upper_bound(i);
					bool uj = is_upper_bound(j);
					update_alpha_status(i);
					update_alpha_status(j);
					int k;
					if(ui != is_upper_bound(i))
					{
						Q_i = get_Q(i,l);
						if(ui)
							for(k=0;k<l;k++)
								G_bar[k] -= C_i * Q_i[k];
						else
							for(k=0;k<l;k++)
								G_bar[k] += C_i * Q_i[k];
						delete[] Q_i;
					}

					if(uj != is_upper_bound(j))
					{
						Q_j = get_Q(j,l);
						if(uj)
							for(k=0;k<l;k++)
								G_bar[k] -= C_j * Q_j[k];
						else
							for(k=0;k<l;k++)
								G_bar[k] += C_j * Q_j[k];
						delete[] Q_j;
					}


				}
			}



			if(iter >= max_iter)
				{
					if(active_size < l)
					{
						// reconstruct the whole gradient to calculate objective value
						reconstruct_gradient();
						active_size = l;
						//info("*");
					}
					fprintf(stderr,"\nWARNING: reaching max number of iterations\n");
				}

				// calculate rho

				si->rho = calculate_rho();

				// calculate objective value
				{
					double v = 0;
					int i;
					for(i=0;i<l;i++)
						v += alpha[i] * (G[i] + p[i]);

					si->obj = v/2;
				}


				double *alpha2 = new double[l];
				// put back the solution
				{
					for(int i=0;i<l;i++)
						alpha2[active_set[i]] = alpha[i];
				}

				delete[] alpha;

				alpha = alpha2;

				// juggle everything back
				/*{
					for(int i=0;i<l;i++)
						while(active_set[i] != i)
							swap_index(i,active_set[i]);
							// or Q.swap_index(i,active_set[i]);
				}*/

				si->upper_bound_p = Cp;
				si->upper_bound_n = Cn;

				//info("\noptimization finished, #iter = %d\n",iter);

				delete[] p;
				delete[] y;
				//delete[] alpha;
				delete[] alpha_status;
				delete[] active_set;
				delete[] G;
				delete[] G_bar;


}
static void solve_epsilon_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha_r, SolutionInfo* si)
{
	int l = prob->l;
	alpha = new double[2*l];
	double *linear_term = new double[2*l];
	//schar *
	y = new schar[2*l];
	int i;

	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		linear_term[i] = param->p - prob->y[i];
		y[i] = 1;

		alpha[i+l] = 0;
		linear_term[i+l] = param->p + prob->y[i];
		y[i+l] = -1;
	}


	init_kernel(*prob,*param);

	p = linear_term;

	Solve(2*l, param->C, param->C, param->eps, si, param->shrinking);

	double sum_alpha = 0;

	for(i=0;i<l;i++)
	{
		alpha_r[i] = alpha[i] - alpha[i+l];
		sum_alpha += fabs(alpha_r[i]);
	}
	//info("nu = %f\n",sum_alpha/(param->C*l));

	delete[] alpha;
	//delete[] linear_term;
	//delete[] y;
}

double kernel_function(int i, int j) {
	return exp(-gamma1*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
}

void delete_kernel() {
	delete[] x;
	delete[] x_square;
	delete[] QD ;
	delete[]	sign ;
	delete[]		indexx ;
}

void init_kernel(const svm_problem& prob, const svm_parameter& param) {
	l = prob.l;
	gamma1=param.gamma;
	clone(x,prob.x,l);
	x_square = new double[l];
			for(int i=0;i<l;i++)
				x_square[i] = dot(x[i],x[i]);

		//	cache = new Cache(l,(long int)(param.cache_size*(1<<20)));
			QD = new double[2*l];
			sign = new schar[2*l];
			indexx = new int[2*l];
			for(int k=0;k<l;k++)
			{
				sign[k] = 1;
				sign[k+l] = -1;
				indexx[k] = k;
				indexx[k+l] = k;
				QD[k] = kernel_function(k,k);
				QD[k+l] = QD[k];
			}
}

static decision_function svm_train_one(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn)
{
	double *alpha1 = Malloc(double,prob->l);
	SolutionInfo si;

		solve_epsilon_svr(prob,param,alpha1,&si);




	// output SVs

	int nSV = 0;
	int nBSV = 0;
	for(int i=0;i<prob->l;i++)
	{
		if(fabs(alpha1[i]) > 0)
		{
			++nSV;
			if(prob->y[i] > 0)
			{
				if(fabs(alpha1[i]) >= si.upper_bound_p)
					++nBSV;
			}
			else
			{
				if(fabs(alpha1[i]) >= si.upper_bound_n)
					++nBSV;
			}
		}
	}

	delete_kernel();

	decision_function f;
	f.alpha = alpha1;
	f.rho = si.rho;
	return f;
}


svm_model *svm_train2(const svm_problem *prob, const svm_parameter *param) {
	gamma1 = param->gamma;


	svm_model *model = Malloc(svm_model,1);
	model->param = *param;
	model->free_sv = 0;	// XXX

	// regression or one-class-svm
	model->nr_class = 2;
	model->label = NULL;
	model->nSV = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->sv_coef = Malloc(double *,1);

	decision_function f = svm_train_one(prob,param,0,0);
			model->rho = Malloc(double,1);
			model->rho[0] = f.rho;

			int nSV = 0;
			int i;
			for(i=0;i<prob->l;i++)
				if(fabs(f.alpha[i]) > 0) ++nSV;
			model->l = nSV;

			model->SV = Malloc(svm_node,nSV);

			model->sv_coef[0] = Malloc(double,nSV);
			model->sv_indices = Malloc(int,nSV);
			int j = 0;
			for(i=0;i<prob->l;i++)
				if(fabs(f.alpha[i]) > 0)
				{
					model->SV[j] = prob->x[i];
					model->sv_coef[0][j] = f.alpha[i];
					model->sv_indices[j] = i+1;
					++j;
				}

			free(f.alpha);

	return model;
}

double k_function(const svm_node *x, const svm_node *y,
		const svm_parameter& param) {

	double sum = 0;

	int dim = min(x->dim, y->dim), i;
	for (i = 0; i < dim; i++) {
		double d = x->values[i] - y->values[i];
		sum += d * d;
	}
	for (; i < x->dim; i++)
		sum += x->values[i] * x->values[i];
	for (; i < y->dim; i++)
		sum += y->values[i] * y->values[i];



	return exp(-param.gamma * sum);
}

double svm_predict(const svm_model *model, const svm_node *x) {
	int nr_class = model->nr_class;
	double *dec_values;

		dec_values = Malloc(double, 1);

	double pred_result = svm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}

double svm_predict_values(const svm_model *model, const svm_node *x,
		double* dec_values) {
	int i;

		double *sv_coef = model->sv_coef[0];
		double sum = 0;

		for (i = 0; i < model->l; i++)

			sum += sv_coef[i] * k_function(x, model->SV + i, model->param);

		sum -= model->rho[0];
		*dec_values = sum;

		if (model->param.svm_type == ONE_CLASS)
			return (sum > 0) ? 1 : -1;
		else
			return sum;

}

