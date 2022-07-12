#define _CRT_SECURE_NO_WARNINGS

using namespace std;
#define PI (atan(1)*4)

//-----------------------------------------------------------------------------
//		Parameters with fixed value 
//-----------------------------------------------------------------------------


#define Lg_r 4.0


#define V_th 0.9
#define Epsilon 1e-10
#define T_ref 1.0


//-----------------------------------------------------------------------------
//		Variables for input parameters
//-----------------------------------------------------------------------------						
int N, NE, NI;                // Number of neurons(total,excitatory,inhibirory)
double T_Max, T_step;         // Total time & time step
double S[4];                  // Coupling strength (E->E,E->I,I->E,I->I) 
int I_CONST;                  // electrode current constant  
double I_const_input;         // constant input current
int random_S;
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
	double t;
	double x;
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
FILE *FP, *FP1, *FP_FFTW, *FP_fire_pattern;
FILE *ffp; // for test

