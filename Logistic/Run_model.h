void update_single_neuron(int id, struct neuron &a, double t,double dt)
{
	double x_old = a.x;

	a.x = Lg_r*x_old*(1-x_old);
	a.t = t + dt;

	if (x_old < V_th && a.x >= V_th && a.t - a.last_fire_time >= T_ref)
	{
		double Dt = a.t - a.last_fire_time;
		a.fire_num++;
		a.last_fire_time = a.t;
		a.if_fired = 1;

		if (record_data[0]) 
		{
			fwrite(&a.t, sizeof(double), 1, FP);
			double s = id;
			fwrite(&s, sizeof(double), 1, FP);
		}

		for (int i = 0; i < N; i++)
		{
			if (i == id)
				continue;
			if (id < NE && i < NE)
				neu[i].wait_strength_E += CS[id][i];
			else if (id < NE && i >= NE)
				neu[i].wait_strength_E += CS[id][i];
			else if (id >= NE && i < NE)
				neu[i].wait_strength_I += CS[id][i];
			else
				neu[i].wait_strength_I += CS[id][i];
		}
	}
}

void update_all_neuron(struct neuron *a, double t)
{
#pragma omp parallel for num_threads(num_threads_openmp)
	for (int id = 0; id < N; id++)
	{
		a[id].if_fired = 0;
		if ((a[id].wait_strength_E == 0) && (a[id].wait_strength_I == 0))
			continue;

		double v_start = a[id].x;

		if (a[id].x + a[id].wait_strength_E - a[id].wait_strength_I < 1 && \
			a[id].x + a[id].wait_strength_E - a[id].wait_strength_I > 0)  // revise
		{
			a[id].x += a[id].wait_strength_E;
			a[id].x -= a[id].wait_strength_I;
		}

		a[id].wait_strength_E = 0;
		a[id].wait_strength_I = 0;


		// check fire 
		if (v_start < V_th && a[id].x >= V_th && t - a[id].last_fire_time >= T_ref)
		{
			a[id].last_fire_time = t;
			a[id].if_fired = 1;
		}
	}

	int continue_update = 0;
	for (int id = 0; id < N; id++)
	{
		if (a[id].if_fired)
		{
			continue_update = 1;
			if (record_data[0])
			{
				fwrite(&t, sizeof(double), 1, FP);
				double s = id;
				fwrite(&s, sizeof(double), 1, FP);
			}
			a[id].fire_num++;
			a[id].last_fire_time = t;

			for (int i = 0; i < N; i++)
			{
				if (i == id)
					continue;
				if (id < NE && i < NE)
					a[i].wait_strength_E += CS[id][i];
				else if (id < NE && i >= NE)
					a[i].wait_strength_E += CS[id][i];
				else if (id >= NE && i < NE)
					a[i].wait_strength_I += CS[id][i];
				else
					a[i].wait_strength_I += CS[id][i];
			}
		}
	}

	if (continue_update)
		update_all_neuron(a, t);
}


void Run_model()
{
	num_threads_openmp = N >= 8 ? 8 : 4;
	double t = 0, tt = 0, tt_fftw = 0, t_lib = 0, t_fp = 0;
	double t_test = 0;

	while (t < T_Max)
	{

// #pragma omp parallel for num_threads(num_threads_openmp)
//! warning: file IO is not thread safe here
		for (int i = 0; i < N; i++)
			update_single_neuron(i, neu[i], t, T_step);
		t += T_step;
		
		update_all_neuron(neu, t);
	

		if (record_data[1] && t > Record_x_start && t <= Record_x_end && t - tt >= 0.01)
		{
			tt = t;
			fwrite(&t, sizeof(double), 1, FP1);
			for (int i = 0; i < N; i++)
				fwrite(&neu[i].x, sizeof(double), 1, FP1);

		}
	
	}
}
