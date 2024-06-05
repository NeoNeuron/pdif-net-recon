#define _CRT_SECURE_NO_WARNINGS

using namespace std;
#define PI (atan(1)*4)

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
#define R_x_th 8.0
#define Epsilon 1e-10
#define T_ref 0.3

double x_th;



//-----------------------------------------------------------------------------
//		Variables for input parameters
//-----------------------------------------------------------------------------
char model[10];				  //(L--Lorenz,R--Rossler)							
int N, NE, NI;                // Number of neurons(total,excitatory,inhibirory)
double T_Max, T_step;         // Total time & time step
double S[4];                  // Coupling strength (E->E,E->I,I->E,I->I) 
int I_CONST;                  // electrode current constant  
double I_const_input;         // constant input current
int full_toggle;			// toggle for full-version config.
int random_S;
double P_c;                   // Connect probability 
int Lyapunov;                     // compute largest lyapunov exponnet
int Power_spectrum = 0;				  // record v for power spectrum	
int record_data[4];				// save data or not
int TrialID = 0;			   // Default: 0.  for multiple trials with fixed CS and change state seeds 
char file[200];				  // Record data path
double Record_x_start, Record_x_end;

//-----------------------------------------------------------------------------
//		Netwrok Information
//-----------------------------------------------------------------------------
double **Connect_Matrix;         // Connect matrix CM_ij, i-->j
double **CS;					 // Coupling strength matrix
struct neuron 
{
	double t;
	double x,dx,y,z;
	double I_input;
	double last_fire_time;
	double fire_num;
	int if_fired;

	long seed;
	double wait_strength_E, wait_strength_I;
	int state;     //1--neu,0--neu_old
};
struct neuron *neu, *neu_old;

//-----------------------------------------------------------------------------
//		Record firing time and voltage
//-----------------------------------------------------------------------------
FILE *FP,*FPx, *FPy, *FPz, *FP_FFTW, *FP_fire_pattern;
FILE *ffp; // for test

//-----------------------------------------------------------------------------
//		Parameters used in RK4 scheme
//-----------------------------------------------------------------------------
double *x_start, *dx_start;
double *w, *w0, **k;

