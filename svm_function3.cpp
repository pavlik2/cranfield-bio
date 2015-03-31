#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#include "svm_function.h"

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
static inline double powi(double base, int times) {
	double tmp = base, ret = 1.0;

	for (int t = times; t > 0; t /= 2) {
		if (t % 2 == 1)
			ret *= tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static void print_string_stdout(const char *s) {
	fputs(s, stdout);
	fflush(stdout);
}
static void (*svm_print_string)(const char *) = &print_string_stdout;
#if DEBUG
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt, ...) {
}
#endif

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//

class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual
	#ifdef FLOAT1g
	float
#else
	double
#endif
 *get_QD() const = 0;

	virtual ~QMatrix() {
	}
};

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//

static double k_function(const svm_node *x, const svm_node *y,
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
	Solver() {
	}
	;
	virtual ~Solver() {
	}
	;

	struct SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
	};

	void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
			double *alpha_, double Cp, double Cn, double eps, SolutionInfo* si,
			int shrinking);
protected:
	int active_size;
	schar *y;
	double *G;		// gradient of objective function
	enum {
		LOWER_BOUND, UPPER_BOUND, FREE
	};
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double *alpha;
	const QMatrix *Q;
	const
#ifdef FLOAT1g
float
#else
double
#endif
 *QD;
	double eps;
	double Cp, Cn;
	double *p;
	int *active_set;
	double *G_bar;		// gradient, if we treat free variables as 0
	int l;
	bool unshrink;	// XXX

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
	bool is_free(int i) {
		return alpha_status[i] == FREE;
	}

	void reconstruct_gradient();
	virtual int select_working_set(int &i, int &j);
	virtual double calculate_rho();

private:
	bool be_shrunk(int i, double Gmax1, double Gmax2);
};

void Solver::reconstruct_gradient() {
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

	if (2 * nr_free < active_size)
		info("\nWARNING: using -h 0 may be faster\n");

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
				double alpha_i = alpha[i];
				for (j = active_size; j < l; j++)
					G[j] += alpha_i * Q_i[j];
			}
	}
}

void Solver::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		double *alpha_, double Cp, double Cn, double eps, SolutionInfo* si,
		int shrinking) {
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
		G = new double[l];
		G_bar = new double[l];
		int i;
		for (i = 0; i < l; i++) {
			G[i] = p[i];
			G_bar[i] = 0;
		}
		for (i = 0; i < l; i++)
			if (!is_lower_bound(i)) {
				const Qfloat *Q_i = Q.get_Q(i, l);
				double alpha_i = alpha[i];
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

			info(".");
		}

		int i, j;
		if (select_working_set(i, j) != 0) {
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = l;
			info("*");
			if (select_working_set(i, j) != 0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}

		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully

		const Qfloat *Q_i = Q.get_Q(i, active_size);
		const Qfloat *Q_j = Q.get_Q(j, active_size);

		double C_i = get_C(i);
		double C_j = get_C(j);

		double old_alpha_i = alpha[i];
		double old_alpha_j = alpha[j];

		if (y[i] != y[j]) {
			double quad_coef = QD[i] + QD[j] + 2 * Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (-G[i] - G[j]) / quad_coef;
			double diff = alpha[i] - alpha[j];
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
			double quad_coef = QD[i] + QD[j] - 2 * Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (G[i] - G[j]) / quad_coef;
			double sum = alpha[i] + alpha[j];
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

		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;

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
			info("*");
		}
		fprintf(stderr, "\nWARNING: reaching max number of iterations\n");
	}

	// calculate rho

	si->rho = calculate_rho();

	// calculate objective value
	{
		double v = 0;
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

	info("\noptimization finished, #iter = %d\n", iter);

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j) {
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
		Q_i = Q->get_Q(i, active_size);

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

	if (Gmax + Gmax2 < eps)
		return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

bool Solver::be_shrunk(int i, double Gmax1, double Gmax2) {
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

double Solver::calculate_rho() {
	double r;
	int nr_free = 0;
	double ub = INF, lb = -INF, sum_free = 0;
	for (int i = 0; i < active_size; i++) {
		double yG = y[i] * G[i];

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
#include "gpu.h"

class SVR_Q: public QMatrix {
public:


	SVR_Q(const svm_problem& prob,  double gamma)

	{
		l = prob.l;
 l2=l*2;
		QD = new
#ifdef FLOAT1g
float
#else
double
#endif
[l2];
		sign = new schar[l2];
		index = new int[l2];

		buf = new Qfloat[l2*l2];

		if (!gpu_t) {

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
											- 2 * dot(x[i], x[j]))));}
			delete[] x_square;

			for (int i = 0; i < l2; i++)
				for (int j = 0; j < l2; j++)
					buf[i * l + j] = (Qfloat) sign[i] * (Qfloat) sign[j]
							* data[index[i] * l +index[j]];
			delete[] data;
		}

	}

	Qfloat *get_Q(int i, int len) const {

		return buf + i * l2;
	}



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



#ifdef FLOAT1g
float
#else
double
#endif
 *get_QD() const {
		return QD;
	}

	~SVR_Q() {

		delete[] sign;
		delete[] index;
		delete[] buf;
		delete[] QD;
	}
private:
	Qfloat *data;
	int l;
	int l2;
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
	SVC_Q(const svm_problem& prob, schar *y, double gamma)

	{

		l = prob.l;
		if (!gpu_t) {
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
		} else {
			QD = new
#ifdef FLOAT1g
float
#else
double
#endif
[l];
			data = new float[l * l];
			RunGPUKERNEL(prob, y,
#ifdef FLOAT1g
					(float)
#endif
					gamma, data, QD);
#ifdef DEBUG
			printf("Kernel launched");
#endif
		}

	}


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

	Qfloat *get_Q(int i, int len) const {

		return data + i * l;
	}


#ifdef FLOAT1g
float
#else
double
#endif
 *get_QD() const {
		return QD;
	}

	~SVC_Q() {

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
static void solve_c_svc(const svm_problem *prob, const svm_parameter* param,
		double *alpha, Solver::SolutionInfo* si, double Cp, double Cn) {
	int l = prob->l;
	double *minus_ones = new double[l];
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

	double sum_alpha = 0;
	for (i = 0; i < l; i++)
		sum_alpha += alpha[i];

	if (Cp == Cn)
		info("nu = %f\n", sum_alpha / (Cp * prob->l));

	for (i = 0; i < l; i++)
		alpha[i] *= y[i];

	delete[] minus_ones;
	delete[] y;
}

//
// decision_function
//
struct decision_function {
	double *alpha;
	double rho;
};

static void solve_epsilon_svr(const svm_problem *prob,
		const svm_parameter *param, double *alpha, Solver::SolutionInfo* si) {
	int l = prob->l;
	double *alpha2 = new double[2 * l];
	double *linear_term = new double[2 * l];
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
	SVR_Q svc = SVR_Q(*prob,  param->gamma);
	s.Solve(2 * l, svc, linear_term, y, alpha2, param->C, param->C, param->eps,
			si, param->shrinking);

	double sum_alpha = 0;
	for (i = 0; i < l; i++) {
		alpha[i] = alpha2[i] - alpha2[i + l];
		sum_alpha += fabs(alpha[i]);
	}
	info("nu = %f\n", sum_alpha / (param->C * l));

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

static decision_function svm_train_one(const svm_problem *prob,
		const svm_parameter *param, double Cp, double Cn) {
	double *alpha = Malloc(double,prob->l);
	Solver::SolutionInfo si;

	switch (param->svm_type) {
	case C_SVC:
		solve_c_svc(prob, param, alpha, &si, Cp, Cn);
		break;

	case EPSILON_SVR:
		solve_epsilon_svr(prob, param, alpha, &si);
		break;

	}

	info("obj = %f, rho = %f\n", si.obj, si.rho);

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

	info("nSV = %d, nBSV = %d\n", nSV, nBSV);

	decision_function f;
	f.alpha = alpha;
	f.rho = si.rho;
	return f;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
static void multiclass_probability(int k, double **r, double *p) {
	int t, j;
	int iter = 0, max_iter = max(100, k);
	double **Q = Malloc(double *,k);
	double *Qp = Malloc(double,k);
	double pQp, eps = 0.005 / k;

	for (t = 0; t < k; t++) {
		p[t] = 1.0 / k;  // Valid if k = 1
		Q[t] = Malloc(double,k);
		Q[t][t] = 0;
		for (j = 0; j < t; j++) {
			Q[t][t] += r[j][t] * r[j][t];
			Q[t][j] = Q[j][t];
		}
		for (j = t + 1; j < k; j++) {
			Q[t][t] += r[j][t] * r[j][t];
			Q[t][j] = -r[j][t] * r[t][j];
		}
	}
	for (iter = 0; iter < max_iter; iter++) {
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp = 0;
		for (t = 0; t < k; t++) {
			Qp[t] = 0;
			for (j = 0; j < k; j++)
				Qp[t] += Q[t][j] * p[j];
			pQp += p[t] * Qp[t];
		}
		double max_error = 0;
		for (t = 0; t < k; t++) {
			double error = fabs(Qp[t] - pQp);
			if (error > max_error)
				max_error = error;
		}
		if (max_error < eps)
			break;

		for (t = 0; t < k; t++) {
			double diff = (-Qp[t] + pQp) / Q[t][t];
			p[t] += diff;
			pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff)
					/ (1 + diff);
			for (j = 0; j < k; j++) {
				Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff);
				p[j] /= (1 + diff);
			}
		}
	}
	if (iter >= max_iter)
		info("Exceeds max_iter in multiclass_prob\n");
	for (t = 0; t < k; t++)
		free(Q[t]);
	free(Q);
	free(Qp);
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret,
		int **label_ret, int **start_ret, int **count_ret, int *perm) {
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
				label = (int *) realloc(label, max_nr_class * sizeof(int));
				count = (int *) realloc(count, max_nr_class * sizeof(int));
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
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param,
		bool gpu) {
	gpu_t = gpu;

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
		model->sv_coef = Malloc(double *,1);

		decision_function f = svm_train_one(prob, param, 0, 0);
		model->rho = Malloc(double,1);
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
		model->sv_coef[0] = Malloc(double,nSV);
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
		if (nr_class == 1)
			info(
					"WARNING: training data in only one class. See README for details.\n");

#ifdef _DENSE_REP
		svm_node *x = Malloc(svm_node,l);
#else
		svm_node **x = Malloc(svm_node *,l);
#endif
		int i;
		for (i = 0; i < l; i++)
			x[i] = prob->x[perm[i]];

		// calculate weighted C

		double *weighted_C = Malloc(double, nr_class);
		for (i = 0; i < nr_class; i++)
			weighted_C[i] = param->C;
		for (i = 0; i < param->nr_weight; i++) {
			int j;
			for (j = 0; j < nr_class; j++)
				if (param->weight_label[i] == label[j])
					break;
			if (j == nr_class)
				fprintf(stderr,
						"WARNING: class label %d specified in weight is not found\n",
						param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models

		bool *nonzero = Malloc(bool,l);
		for (i = 0; i < l; i++)
			nonzero[i] = false;
		decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

		double *probA = NULL, *probB = NULL;
		if (param->probability) {
			probA = Malloc(double,nr_class*(nr_class-1)/2);
			probB = Malloc(double,nr_class*(nr_class-1)/2);
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
				sub_prob.y = Malloc(double,sub_prob.l);
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

		model->rho = Malloc(double,nr_class*(nr_class-1)/2);
		for (i = 0; i < nr_class * (nr_class - 1) / 2; i++)
			model->rho[i] = f[i].rho;

		if (param->probability) {
			model->probA = Malloc(double,nr_class*(nr_class-1)/2);
			model->probB = Malloc(double,nr_class*(nr_class-1)/2);
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

		info("Total nSV = %d\n", total_sv);

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

		model->sv_coef = Malloc(double *,nr_class-1);
		for (i = 0; i < nr_class - 1; i++)
			model->sv_coef[i] = Malloc(double,total_sv);

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

// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param,
		int nr_fold, double *target) {
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int,l);
	int nr_class;
	if (nr_fold > l) {
		nr_fold = l;
		fprintf(stderr,
				"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if ((param->svm_type == C_SVC || param->svm_type == NU_SVC)
			&& nr_fold < l) {
		int *start = NULL;
		int *label = NULL;
		int *count = NULL;
		svm_group_classes(prob, &nr_class, &label, &start, &count, perm);

		// random shuffle and then data grouped by fold using the array perm
		int *fold_count = Malloc(int,nr_fold);
		int c;
		int *index = Malloc(int,l);
		for (i = 0; i < l; i++)
			index[i] = perm[i];
		for (c = 0; c < nr_class; c++)
			for (i = 0; i < count[c]; i++) {
				int j = i + rand() % (count[c] - i);
				swap(index[start[c] + j], index[start[c] + i]);
			}
		for (i = 0; i < nr_fold; i++) {
			fold_count[i] = 0;
			for (c = 0; c < nr_class; c++)
				fold_count[i] += (i + 1) * count[c] / nr_fold
						- i * count[c] / nr_fold;
		}
		fold_start[0] = 0;
		for (i = 1; i <= nr_fold; i++)
			fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
		for (c = 0; c < nr_class; c++)
			for (i = 0; i < nr_fold; i++) {
				int begin = start[c] + i * count[c] / nr_fold;
				int end = start[c] + (i + 1) * count[c] / nr_fold;
				for (int j = begin; j < end; j++) {
					perm[fold_start[i]] = index[j];
					fold_start[i]++;
				}
			}
		fold_start[0] = 0;
		for (i = 1; i <= nr_fold; i++)
			fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
		free(start);
		free(label);
		free(count);
		free(index);
		free(fold_count);
	} else {
		for (i = 0; i < l; i++)
			perm[i] = i;
		for (i = 0; i < l; i++) {
			int j = i + rand() % (l - i);
			swap(perm[i], perm[j]);
		}
		for (i = 0; i <= nr_fold; i++)
			fold_start[i] = i * l / nr_fold;
	}

	for (i = 0; i < nr_fold; i++) {
		int begin = fold_start[i];
		int end = fold_start[i + 1];
		int j, k;
		struct svm_problem subprob;

		subprob.l = l - (end - begin);
#ifdef _DENSE_REP
		subprob.x = Malloc(struct svm_node,subprob.l);
#else
		subprob.x = Malloc(struct svm_node*,subprob.l);
#endif
		subprob.y = Malloc(double,subprob.l);

		k = 0;
		for (j = 0; j < begin; j++) {
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for (j = end; j < l; j++) {
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct svm_model *submodel = svm_train(&subprob, param, gpu_t);
		if (param->probability
				&& (param->svm_type == C_SVC || param->svm_type == NU_SVC)) {
			double *prob_estimates = Malloc(double,svm_get_nr_class(submodel));
			for (j = begin; j < end; j++)
#ifdef _DENSE_REP
				target[perm[j]] = svm_predict_probability(submodel,
						(prob->x + perm[j]), prob_estimates);
#else
			target[perm[j]] = svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
#endif
			free(prob_estimates);
		} else
			for (j = begin; j < end; j++)
#ifdef _DENSE_REP
				target[perm[j]] = svm_predict(submodel, prob->x + perm[j]);
#else
		target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
#endif
		svm_free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}

int svm_get_svm_type(const svm_model *model) {
	return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model) {
	return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label) {
	if (model->label != NULL)
		for (int i = 0; i < model->nr_class; i++)
			label[i] = model->label[i];
}

void svm_get_sv_indices(const svm_model *model, int* indices) {
	if (model->sv_indices != NULL)
		for (int i = 0; i < model->l; i++)
			indices[i] = model->sv_indices[i];
}

int svm_get_nr_sv(const svm_model *model) {
	return model->l;
}

double svm_get_svr_probability(const svm_model *model) {
	if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR)
			&& model->probA != NULL)
		return model->probA[0];
	else {
		fprintf(stderr,
				"Model doesn't contain information for SVR probability inference\n");
		return 0;
	}
}
static double sigmoid_predict(double decision_value, double A, double B) {
	double fApB = decision_value * A + B;
	// 1-p used later; avoid catastrophic cancellation
	if (fApB >= 0)
		return exp(-fApB) / (1.0 + exp(-fApB));
	else
		return 1.0 / (1 + exp(fApB));
}

double svm_predict_values(const svm_model *model, const svm_node *x,
		double* dec_values) {
	int i;

	if(model->param.svm_type == ONE_CLASS ||
		   model->param.svm_type == EPSILON_SVR ||
		   model->param.svm_type == NU_SVR)
		{
			double *sv_coef = model->sv_coef[0];
			double sum = 0;

			for(i=0;i<model->l;i++)

				sum += sv_coef[i] * k_function(x,model->SV+i,model->param);

			sum -= model->rho[0];
			*dec_values = sum;


				return sum;
		}
		else
		{


	int nr_class = model->nr_class;
	int l = model->l;

	double *kvalue = Malloc(double,l);
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
			double sum = 0;
			int si = start[i];
			int sj = start[j];
			int ci = model->nSV[i];
			int cj = model->nSV[j];

			int k;
			double *coef1 = model->sv_coef[j - 1];
			double *coef2 = model->sv_coef[i];
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

double svm_predict(const svm_model *model, const svm_node *x) {
	int nr_class = model->nr_class;
	double *dec_values;
	if(model->param.svm_type == ONE_CLASS ||
		   model->param.svm_type == EPSILON_SVR ||
		   model->param.svm_type == NU_SVR)
			dec_values = Malloc(double, 1);
		else
	dec_values = Malloc(double, nr_class*(nr_class-1)/2);
	double pred_result = svm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}

double svm_predict_probability(const svm_model *model, const svm_node *x,
		double *prob_estimates) {
	if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC)
			&& model->probA != NULL && model->probB != NULL) {
		int i;
		int nr_class = model->nr_class;
		double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
		svm_predict_values(model, x, dec_values);

		double min_prob = 1e-7;
		double **pairwise_prob = Malloc(double *,nr_class);
		for (i = 0; i < nr_class; i++)
			pairwise_prob[i] = Malloc(double,nr_class);
		int k = 0;
		for (i = 0; i < nr_class; i++)
			for (int j = i + 1; j < nr_class; j++) {
				pairwise_prob[i][j] = min(
						max(
								sigmoid_predict(dec_values[k], model->probA[k],
										model->probB[k]), min_prob),
						1 - min_prob);
				pairwise_prob[j][i] = 1 - pairwise_prob[i][j];
				k++;
			}
		multiclass_probability(nr_class, pairwise_prob, prob_estimates);

		int prob_max_idx = 0;
		for (i = 1; i < nr_class; i++)
			if (prob_estimates[i] > prob_estimates[prob_max_idx])
				prob_max_idx = i;
		for (i = 0; i < nr_class; i++)
			free(pairwise_prob[i]);
		free(dec_values);
		free(pairwise_prob);
		return model->label[prob_max_idx];
	} else
		return svm_predict(model, x);
}

static const char *svm_type_table[] = { "c_svc", "nu_svc", "one_class",
		"epsilon_svr", "nu_svr", NULL };

static const char *kernel_type_table[] = { "linear", "polynomial", "rbf",
		"sigmoid", "precomputed", NULL };

int svm_save_model(const char *model_file_name, const svm_model *model) {
	FILE *fp = fopen(model_file_name, "w");
	if (fp == NULL)
		return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	const svm_parameter& param = model->param;

	fprintf(fp, "svm_type %s\n", svm_type_table[param.svm_type]);
	fprintf(fp, "kernel_type %s\n", kernel_type_table[param.kernel_type]);

	if (param.kernel_type == POLY)
		fprintf(fp, "degree %d\n", param.degree);

	if (param.kernel_type == POLY || param.kernel_type == RBF
			|| param.kernel_type == SIGMOID)
		fprintf(fp, "gamma %g\n", param.gamma);

	if (param.kernel_type == POLY || param.kernel_type == SIGMOID)
		fprintf(fp, "coef0 %g\n", param.coef0);

	int nr_class = model->nr_class;
	int l = model->l;
	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "total_sv %d\n", l);

	{
		fprintf(fp, "rho");
		for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
			fprintf(fp, " %g", model->rho[i]);
		fprintf(fp, "\n");
	}

	if (model->label) {
		fprintf(fp, "label");
		for (int i = 0; i < nr_class; i++)
			fprintf(fp, " %d", model->label[i]);
		fprintf(fp, "\n");
	}

	if (model->probA) // regression has probA only
	{
		fprintf(fp, "probA");
		for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
			fprintf(fp, " %g", model->probA[i]);
		fprintf(fp, "\n");
	}
	if (model->probB) {
		fprintf(fp, "probB");
		for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
			fprintf(fp, " %g", model->probB[i]);
		fprintf(fp, "\n");
	}

	if (model->nSV) {
		fprintf(fp, "nr_sv");
		for (int i = 0; i < nr_class; i++)
			fprintf(fp, " %d", model->nSV[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "SV\n");
	const double * const *sv_coef = model->sv_coef;
#ifdef _DENSE_REP
	const svm_node *SV = model->SV;
#else
	const svm_node * const *SV = model->SV;
#endif

	for (int i = 0; i < l; i++) {
		for (int j = 0; j < nr_class - 1; j++)
			fprintf(fp, "%.16g ", sv_coef[j][i]);

#ifdef _DENSE_REP
		const svm_node *p = (SV + i);

		if (param.kernel_type == PRECOMPUTED)
			fprintf(fp, "0:%d ", (int) (p->values[0]));
		else
			for (int j = 0; j < p->dim; j++)
				if (p->values[j] != 0.0)
					fprintf(fp, "%d:%.8g ", j, p->values[j]);
#else
		const svm_node *p = SV[i];

		if(param.kernel_type == PRECOMPUTED)
		fprintf(fp,"0:%d ",(int)(p->value));
		else
		while(p->index != -1)
		{
			fprintf(fp,"%d:%.8g ",p->index,p->value);
			p++;
		}
#endif
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return -1;
	else
		return 0;
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input) {
	int len;

	if (fgets(line, max_line_len, input) == NULL)
		return NULL;

	while (strrchr(line, '\n') == NULL) {
		max_line_len *= 2;
		line = (char *) realloc(line, max_line_len);
		len = (int) strlen(line);
		if (fgets(line + len, max_line_len - len, input) == NULL)
			break;
	}
	return line;
}

svm_model *svm_load_model(const char *model_file_name) {
	FILE *fp = fopen(model_file_name, "rb");
	if (fp == NULL)
		return NULL;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	// read parameters

	svm_model *model = Malloc(svm_model,1);
	svm_parameter& param = model->param;
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->sv_indices = NULL;
	model->label = NULL;
	model->nSV = NULL;

	char cmd[81];
	while (1) {
		fscanf(fp, "%80s", cmd);

		if (strcmp(cmd, "svm_type") == 0) {
			fscanf(fp, "%80s", cmd);
			int i;
			for (i = 0; svm_type_table[i]; i++) {
				if (strcmp(svm_type_table[i], cmd) == 0) {
					param.svm_type = i;
					break;
				}
			}
			if (svm_type_table[i] == NULL) {
				fprintf(stderr, "unknown svm type.\n");

				setlocale(LC_ALL, old_locale);
				free(old_locale);
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		} else if (strcmp(cmd, "kernel_type") == 0) {
			fscanf(fp, "%80s", cmd);
			int i;
			for (i = 0; kernel_type_table[i]; i++) {
				if (strcmp(kernel_type_table[i], cmd) == 0) {
					param.kernel_type = i;
					break;
				}
			}
			if (kernel_type_table[i] == NULL) {
				fprintf(stderr, "unknown kernel function.\n");

				setlocale(LC_ALL, old_locale);
				free(old_locale);
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		} else if (strcmp(cmd, "degree") == 0)
			fscanf(fp, "%d", &param.degree);
		else if (strcmp(cmd, "gamma") == 0)
			fscanf(fp, "%lf", &param.gamma);
		else if (strcmp(cmd, "coef0") == 0)
			fscanf(fp, "%lf", &param.coef0);
		else if (strcmp(cmd, "nr_class") == 0)
			fscanf(fp, "%d", &model->nr_class);
		else if (strcmp(cmd, "total_sv") == 0)
			fscanf(fp, "%d", &model->l);
		else if (strcmp(cmd, "rho") == 0) {
			int n = model->nr_class * (model->nr_class - 1) / 2;
			model->rho = Malloc(double,n);
			for (int i = 0; i < n; i++)
				fscanf(fp, "%lf", &model->rho[i]);
		} else if (strcmp(cmd, "label") == 0) {
			int n = model->nr_class;
			model->label = Malloc(int,n);
			for (int i = 0; i < n; i++)
				fscanf(fp, "%d", &model->label[i]);
		} else if (strcmp(cmd, "probA") == 0) {
			int n = model->nr_class * (model->nr_class - 1) / 2;
			model->probA = Malloc(double,n);
			for (int i = 0; i < n; i++)
				fscanf(fp, "%lf", &model->probA[i]);
		} else if (strcmp(cmd, "probB") == 0) {
			int n = model->nr_class * (model->nr_class - 1) / 2;
			model->probB = Malloc(double,n);
			for (int i = 0; i < n; i++)
				fscanf(fp, "%lf", &model->probB[i]);
		} else if (strcmp(cmd, "nr_sv") == 0) {
			int n = model->nr_class;
			model->nSV = Malloc(int,n);
			for (int i = 0; i < n; i++)
				fscanf(fp, "%d", &model->nSV[i]);
		} else if (strcmp(cmd, "SV") == 0) {
			while (1) {
				int c = getc(fp);
				if (c == EOF || c == '\n')
					break;
			}
			break;
		} else {
			fprintf(stderr, "unknown text in model file: [%s]\n", cmd);

			setlocale(LC_ALL, old_locale);
			free(old_locale);
			free(model->rho);
			free(model->label);
			free(model->nSV);
			free(model);
			return NULL;
		}
	}

	// read sv_coef and SV

	int elements = 0;
	long pos = ftell(fp);

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	char *p, *endptr, *idx, *val;

#ifdef _DENSE_REP
	int max_index = 1;
	// read the max dimension of all vectors
	while (readline(fp) != NULL) {
		char *p;
		p = strrchr(line, ':');
		if (p != NULL) {
			while (*p != ' ' && *p != '\t' && p > line)
				p--;
			if (p > line)
				max_index = (int) strtol(p, &endptr, 10) + 1;
		}
		if (max_index > elements)
			elements = max_index;
	}
#else
	while(readline(fp)!=NULL)
	{
		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
			break;
			++elements;
		}
	}
	elements += model->l;

#endif
	fseek(fp, pos, SEEK_SET);

	int m = model->nr_class - 1;
	int l = model->l;
	model->sv_coef = Malloc(double *,m);
	int i;
	for (i = 0; i < m; i++)
		model->sv_coef[i] = Malloc(double,l);

#ifdef _DENSE_REP
	int index;
	model->SV = Malloc(svm_node,l);

	for (i = 0; i < l; i++) {
		readline(fp);

		model->SV[i].values = Malloc(double, elements);
		model->SV[i].dim = 0;

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p, &endptr);
		for (int k = 1; k < m; k++) {
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p, &endptr);
		}

		int *d = &(model->SV[i].dim);
		while (1) {
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;
			index = (int) strtol(idx, &endptr, 10);
			while (*d < index)
				model->SV[i].values[(*d)++] = 0.0;
			model->SV[i].values[(*d)++] = strtod(val, &endptr);
		}
	}
#else
	model->SV = Malloc(svm_node*,l);
	svm_node *x_space = NULL;
	if(l>0) x_space = Malloc(svm_node,elements);
	int j=0;
	for(i=0;i<l;i++)
	{
		readline(fp);
		model->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p,&endptr);
		}

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
			break;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			x_space[j].value = strtod(val,&endptr);

			++j;
		}
		x_space[j++].index = -1;
	}
#endif
	free(line);

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	model->free_sv = 1;	// XXX
	return model;
}

void svm_free_model_content(svm_model* model_ptr) {
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

int svm_check_probability_model(const svm_model *model) {
	return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC)
			&& model->probA != NULL && model->probB != NULL)
			|| ((model->param.svm_type == EPSILON_SVR
					|| model->param.svm_type == NU_SVR) && model->probA != NULL);
}

void svm_set_print_string_function(void (*print_func)(const char *)) {
	if (print_func == NULL)
		svm_print_string = &print_string_stdout;
	else
		svm_print_string = print_func;
}
