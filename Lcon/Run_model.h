
void Exchange(struct neuron &a, struct neuron b)  // a <-- b
{
	a.t = b.t;
	a.x = b.x;
	a.dx = b.dx;
	a.y = b.y;
	a.z = b.z;

	a.I_input = b.I_input;
	a.last_fire_time = b.last_fire_time;
	a.if_fired = 0;
}

void Run_model()
{
	double t = 0, tt = 0, tt_fftw = 0, t_lib = 0, t_fp = 0;
	double t_test = 0;

	while (t < T_Max)
	{
		Update_RK4(N, neu, t, T_step);
		t += T_step;

		if (record_data[1] && t > Record_x_start && t <= Record_x_end && t - tt >= 0.01-1e-4)
		{
			tt = t;
			fwrite(&t, sizeof(double), 1, FP1);
			for (int i = 0; i < N; i++)
				fwrite(&neu[i].x, sizeof(double), 1, FP1);
		}
	}
}
