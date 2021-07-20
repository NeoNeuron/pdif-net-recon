#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <cstring>
#include <malloc.h>
#include <algorithm>
#include <omp.h>

using namespace std;
#define PI (atan(1)*4)
#define Epsilon 1e-10
int num_threads_openmp;

//-----------------------------------------------------------------------------
//		Parameters with fixed value 
//-----------------------------------------------------------------------------

#define sigma 10.0
#define beta 8.0/3
#define rho 28.0

#define R_a 0.2
#define R_b 0.2
#define R_c 5.7


#define L_x_th 10.0
//#define R_x_th 8.0
#define T_ref 0.5

double x_th;



//-----------------------------------------------------------------------------
//		Variables for input parameters
//-----------------------------------------------------------------------------						
int N, NE, NI;                // Number of neurons(total,excitatory,inhibirory)
double T_Max, T_step;         // Total time & time step
double S[4];                  // Coupling strength (E->E,E->I,I->E,I->I) 
int I_CONST;                  // electrode current constant  
double I_const_input;         // constant input current
double Nu, f[2];                // Feedforward Poisson rate and strength(E,I)
int random_S, random_Nu;
double P_c;                   // Connect probability 
int Lyapunov;                     // compute largest lyapunov exponnet
int Power_spectrum = 0;				  // record v for power spectrum	
int record_data[2];				// save data or not
char file[200];				  // Record data path
double Record_x_start, Record_x_end;

//-----------------------------------------------------------------------------
//		Netwrok Information
//-----------------------------------------------------------------------------
double **Connect_Matrix;         // Connect matrix
double **CS;					 // Coupling strength matrix
struct neuron 
{
	double t,Nu;
	double x,dx,y,z;
	double I_input;
	double last_fire_time;
	double fire_num;
	int if_fired;

	double *Poisson_input_time;
	int Poisson_input_num;
	long seed;
	double wait_strength_E, wait_strength_I;
	int state;     //1--neu,0--neu_old
};
struct neuron *neu, *neu_old;


#define MIN(a,b)  ((a)<(b)?(a):(b))
#define MAX(a,b)  ((a)>(b)?(a):(b))
//-----------------------------------------------------------------------------
//		Record firing time and voltage
//-----------------------------------------------------------------------------
FILE *FP,*FP1, *FP_FFTW, *FP_fire_pattern;
FILE *ffp; // for test

