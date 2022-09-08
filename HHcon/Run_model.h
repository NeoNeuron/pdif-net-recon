
double g_f(int n, double t)     //    Sinusoidal input
{
	double s,s0;
	s = f[n]*0.5;
	s0 = (Omega*t + (n + 0.0) / N) * 2 * PI;
	s *= sin(s0) + 1;
	return s;
}

void Generate_Poisson_times(struct neuron &a, double t, double dt)
{
	int k = a.Poisson_input_num;
	double t1;
	if (k == -1)
		t1 = -log(1 - Random(a.seed)) / a.Nu; 
	else
		t1 = a.Poisson_input_time[k];   

	k = 0;
	a.Poisson_input_time[k] = t1;
	while (t1 <= t + dt)
	{
		k++;
		t1 += -log(1 - Random(a.seed)) / a.Nu;    // the last one is larger than t+dt

		if (k + 1 > int(T_Step_Large*a.Nu * 2) + 5)
			a.Poisson_input_time = (double *)realloc(a.Poisson_input_time, (k + 15) * sizeof(double));
		a.Poisson_input_time[k] = t1;
	}
	a.Poisson_input_num = k;
}

void Record_Power_spectrum(double t)
{
	fwrite(&t,sizeof(double),1,FP_FFTW);
	for (int i = 0; i < N; i++)
	{
		fwrite(&neu[i].v, sizeof(double), 1, FP_FFTW);
	}
}


void Run_model()
{
	num_threads_openmp = N >= 8 ? 8 : 4;

	if(ode_type !=2 && ode_type !=4)
	{
		printf("\nError ode_type = %d!!!\n", ode_type);
		exit(0);
	}

	double t = 0, tt = 0, tt_fftw = 0, t_lib = 0, t_fp = 0;
	double s = -100, ss = neu[0].v;

	while (t < T_Max)
	{
		if (Nu > Epsilon)
			for (int i = 0; i < N; i++)
				Generate_Poisson_times(neu[i], t, T_Step_Large);

		if(ode_type == 4)
			Update_RK4(N, neu, t, T_Step_Large);
		else
			Update_RK2(N, neu, t, T_Step_Large);



		t += T_Step_Large;

		if (record_data[1] && t > Record_v_start && t <= Record_v_end && t - tt >= 0.5-1e-8)
		{
			tt = t;
			if (!(t - neu[0].last_fire_time <= T_ref && Lib_method))
			{
				fwrite(&t, sizeof(double), 1, FP1);
				for (int id = 0; id < N; id++)
					fwrite(&neu[id].v, sizeof(double), 1, FP1);

				//fwrite(&neu[1].v, sizeof(double), 1, FP1);
				//fwrite(&neu[2].v, sizeof(double), 1, FP1);
			}
		}

		//printf("t=%f v=%f\n",t,neu[0].v);

	}



}
