
/* Lorenz model: Network (Library and Regular)*/
//-----------------------------------------------------------------------------
//		Comments
//-----------------------------------------------------------------------------


#include "Def.h"
#include "Random.h"
#include "Read_parameters.h"
#include "Initialization.h"
#include "Find_cubic_hermite_root.h"
#include "Runge_Kutta4.h"
#include "Run_model.h"
#include "Largest_Lyapunov.h"
#include "Delete.h"





int main(int argc,char **argv)
{
	long seed, seed0, seed1, seed2;
	clock_t t0, t1;	 
	char str[200];
	double MLE;
	double mean_fire_rate;
 
	Read_parameters(seed, seed1);
	if (argc > 1)
	{
		//strcpy(model, argv[1]);
		P_c = atof(argv[1]);
		S[0] = atof(argv[2]); S[1] = S[0]; S[2] = S[0]; S[3] = S[0];
		f = atof(argv[3]); 
		Nu = atof(argv[4]);
	}
	
	/////////////////////	
	out_put_filename();
	seed0 = seed;    // Create connect matrix
	seed2 = seed1;  // Initialization & Poisson
	Initialization(seed0, seed2);


//	x_th = atof(argv[5]);   //revise

	//neu[0].x = 10;   ///EPSP
	//neu[1].x = 0;
	//CS[1][2] = 0;
	//Exchange(neu[2], neu[1]);

	t0 = clock();
	if (Lyapunov)
		MLE = Largest_Lyapunov(seed2, 1, T_step);
	else
		Run_model();
	t1 = clock();


	int total_fire_num[2] = { 0 };
	for (int i = 0; i < N; i++)
		total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;

	
	mean_fire_rate = (total_fire_num[0] + total_fire_num[1]) / T_Max * 1000 / N; //(Hz)
	printf("mean firing rate = %0.2f(Hz)\n", mean_fire_rate);

	printf("Total time = %0.3fs \n\n", double(t1 - t0) / CLOCKS_PER_SEC);
	Delete();
}


////////////////////////// scan
//int main(int argc, char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200], c[10];
//	double MLE;
//	double mean_fire_rate;
//
//	Read_parameters(seed, seed1);
//
//	printf("Run model name: %s\n", argv[1]); // model
//	strcpy(model, argv[1]);
//	P_c = atof(argv[2]);
//	S[0] = atof(argv[3]); S[1] = S[0]; S[2] = S[0]; S[3] = S[0];
//	f = atof(argv[4]); 
//	Nu = atof(argv[5]);
//	double ds = atof(argv[6]);
//
//	/////////////////////
//	Lyapunov = 0;
//	record_data[0] = 1;
//	record_data[1] = 0;
//	Power_spectrum = 0;
//
//	for (int id = atoi(argv[7]); id <= atoi(argv[8]); id++)
//	{
//		S[0] = id * ds;
//
//		out_put_filename();
//		seed0 = seed;    // Create connect matrix
//		seed2 = seed1;  // Initialization & Poisson
//		Initialization(seed0, seed2);
//
//
//		t0 = clock();
//		Run_model();
//		t1 = clock();
//
//		int total_fire_num[2] = { 0 };
//		for (int i = 0; i < N; i++)
//			total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;
//
//		if (NE == N)
//			printf("mean rate(Hz) = %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE);
//		else if (NI == N)
//			printf("mean rate(Hz) = %0.3f\n", total_fire_num[1] / neu[0].t * 1000 / NI);
//		else
//			printf("mean rate(Hz) = %0.3f %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE, total_fire_num[1] / neu[0].t * 1000 / NI);
//
//
//		double rate = total_fire_num[0] / neu[0].t * 1000 / NE;
//
//		printf("s=%0.3f ", S[0]);
//		printf("Total time = %0.3fs\n\n", double(t1 - t0) / CLOCKS_PER_SEC);
//		Delete();
//	}
//
//}
