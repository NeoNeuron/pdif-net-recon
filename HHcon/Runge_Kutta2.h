

void Update_RK2(int n, struct neuron *a, double t, double dt)
{
	double g_f(int n, double t);
	void Update_once_RK4(double *k, double *w, int n, double t, double h);

	//double v_start = a.v, dv_start = a.dv;
	//double w[10], w0[10];  // v,m,h,n,G_se,G_sse,G_si,G_ssi,G_f,G_ff
	//double k[2][4];      // v,m,h,n

	double s_d_e, s_d_i, s_r_e, s_r_i;

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
		w0[i] = w[i] + k[0][i];
	Update_once_RK4(k[1], w0, n, t + dt, dt);   //2

	for (int i = 0; i < 8 * n; i++)
		w[i] = w[i] + (k[0][i] + k[1][i]) / 2.0;


//#pragma omp parallel for num_threads(num_threads_openmp)
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
			a[i].G_sse += f[i] * neu[i].Poisson_input_num;

	}
	Call_num++;
}

