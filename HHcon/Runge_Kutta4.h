double g(double V)
{
	//if (V <= 10)
	//	return 0;
	//else if (V >= 30)
	//	return 1;
	//else
	//	return 1.0 / (1 + exp(-(V - 20) / 2));

	return 1.0 / (1 + exp(-(V - 20) / 2));
}


void Update_once_RK4(double *k, double *w, int n, double t, double h)
{
	double g_f(int n, double t);

#pragma omp parallel for num_threads(num_threads_openmp)
	for (int i = 0; i < N; i++)
	{
		int id = 8 * i;
		double G_f, I_input;
		if (I_CONST)
			I_input = I_const_input;
		else
		{
			G_f = Nu < Epsilon ? g_f(n, t) : w[id+8];
			I_input = -w[id + 4] * (w[id + 0] - V_G_E);
			I_input -= w[id + 6] * (w[id + 0] - V_G_I);
		}

		k[id + 0] = -G_Na * w[id + 1] * w[id + 1] * w[id + 1] * w[id + 2] * (w[id + 0] - E_Na)
			- G_K * w[id + 3] * w[id + 3] * w[id + 3] * w[id + 3] * (w[id + 0] - E_K) - G_L * (w[id + 0] - E_L) + I_input;

		k[id + 0] = k[id + 0] * h / C;																  // v
		k[id + 1] = h * (alpha_m(w[id + 0]) - w[id + 1] * (alpha_m(w[id + 0]) + beta_m(w[id + 0])));  // m
		k[id + 2] = h * (alpha_h(w[id + 0]) - w[id + 2] * (alpha_h(w[id + 0]) + beta_h(w[id + 0])));  // h
		k[id + 3] = h * (alpha_n(w[id + 0]) - w[id + 3] * (alpha_n(w[id + 0]) + beta_n(w[id + 0])));  // n
		k[id + 4] = h * (-w[id + 4] / Sigma_r_E + w[id + 5]);			 //G_se

		double se = 0, si = 0;
		for (int j = 0; j < NE; j++)
		{
			if (CS[j][i] != 0)
				se += CS[j][i] * g(w[8 * j]); //j-->i
		}
		for (int j = NE; j < N; j++)
		{
			if (CS[j][i] != 0)
				si += CS[j][i] * g(w[8 * j]); //j-->i
		}

		k[id + 5] = h * (-w[id + 5] / Sigma_d_E + se);					 //G_sse
		k[id + 6] = h * (-w[id + 6] / Sigma_r_I + w[id + 7]);			 //G_si
		k[id + 7] = h * (-w[id + 7] / Sigma_d_I + si);					 //G_ssi
	}

}

//-----------------------------------------------------------------------------
//		Runge Kutta_4 mehtod: update neuron during determinsteristic parts(i.e. no outside input S and F)
//  	start from time t, end with time t+dt. There are four times to update,
//		so can be simplyfied by one function.
//
//		We just update v,m,h,n with RK4 method, since conductance parts are correct solutions.
//-----------------------------------------------------------------------------

void Update_RK4(int n, struct neuron *a, double t, double dt)
{
	double g_f(int n, double t);

	//double v_start = a.v, dv_start = a.dv;
	//double w[10], w0[10];  // v,m,h,n,G_se,G_sse,G_si,G_ssi
	//double k[4][4];      // v,m,h,n

	double s_d_e, s_d_i,s_r_e,s_r_i;

	for (int i = 0; i < n; i++)
	{
		v_start[i] = a[i].v;
		dv_start[i] = a[i].dv;
		w[i * 8 + 0] = a[i].v;
		w[i * 8 + 1] = a[i].m;
		w[i * 8 + 2] = a[i].h;
		w[i * 8 + 3] = a[i].n;
		w[i * 8 + 4] = a[i].G_se;
		w[i * 8 + 5] = a[i].G_sse;
		w[i * 8 + 6] = a[i].G_si;
		w[i * 8 + 7] = a[i].G_ssi;
		a[i].if_fired = 0;
	}



	for (int i = 0; i < 8 * n; i++)
		w0[i] = w[i];
	Update_once_RK4(k[0], w0, n, t, dt);            //1	
	
	for (int i = 0; i < 8 * n; i++)
		w0[i] = w[i] + 0.5*k[0][i];
	Update_once_RK4(k[1], w0, n, t + dt / 2, dt);   //2


	for (int i = 0; i < 8 * n; i++)
		w0[i] = w[i] + 0.5*k[1][i];
	Update_once_RK4(k[2], w0, n, t + dt / 2, dt);   //3


	for (int i = 0; i < 8 * n; i++)
		w0[i] = w[i] + k[2][i];
	Update_once_RK4(k[3], w0, n, t + dt, dt);   //4

	for (int i = 0; i < 8 * n; i++)
		w[i] = w[i] + (k[0][i] + k[1][i] * 2 + k[2][i] * 2 + k[3][i]) / 6.0;
	

	for (int i = 0; i < n; i++)
	{
		int id = 8 * i;
		a[i].v = w[id + 0], a[i].m = w[id + 1], a[i].h = w[id + 2], a[i].n = w[id + 3];
		a[i].G_se = w[id + 4], a[i].G_sse = w[id + 5], a[i].G_si = w[id + 6], a[i].G_ssi = w[id + 7];
		a[i].t = t + dt;

		if (I_CONST)
			a[i].I_input = I_const_input;
		else
		{
			a[i].I_input = -a[i].G_se * (a[i].v - V_G_E);
			a[i].I_input -= a[i].G_si * (a[i].v - V_G_I);
		}
		a[i].dv = -G_Na * a[i].m * a[i].m * a[i].m * a[i].h * (a[i].v - E_Na)
			- G_K * a[i].n * a[i].n * a[i].n * a[i].n * (a[i].v - E_K) - G_L * (a[i].v - E_L) + a[i].I_input;
		a[i].dv /= C;

		if (v_start[i] < V_th && a[i].v >= V_th && t + dt - a[i].last_fire_time >= T_ref)
		{
			a[i].last_fire_time = cubic_hermite_real_root(t, t + dt, v_start[i], a[i].v, dv_start[i], a[i].dv, V_th);
			a[i].fire_num++;
			a[i].if_fired = 1;
			if (a[i].v < V_th)
				a[i].v = V_th;

			if (record_data[0])
			{
				fwrite(&a[i].last_fire_time, sizeof(double), 1, FP);
				double s = i;
				fwrite(&s, sizeof(double), 1, FP);
			}
		}

		if (abs(a[i].v) > 1e3)
		{
			printf("\nError! Too large time step %f in RK4\n", T_Step_Large);
			printf("i=%d dt=%0.2e last=%f t=%f v=%0.2e dv=%0.2e state=%d\n\n", i, dt, a[i].last_fire_time, a[i].t, a[i].v, a[i].dv, a[i].state);
			getchar();// system("pause");
			//a.v = -65; a.dv = 0;
			exit(1);
		}

		if (Nu > Epsilon)
			a[i].G_sse += f[i]*neu[i].Poisson_input_num;

	}
	Call_num++;
}



