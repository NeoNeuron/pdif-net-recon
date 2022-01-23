
void Update_once_RK4(double *k, double *w, int n, double t, double h)
{
	double I_input;
	if (I_CONST)
		I_input = I_const_input;
	else
		I_input = 0;

	//if (model[0] == 'L')
	//{
	//	k[0] = h * (sigma*(w[1] - w[0]) + I_input);
	//	k[1] = h * (rho*w[0] - w[1] - w[0] * w[2]);
	//	k[2] = h * (-beta * w[2] + w[0] * w[1]);
	//}
	//else
	//{
	//	k[0] = h * (-w[1] - w[2] + I_input);
	//	k[1] = h * (w[0] + R_a * w[1]);
	//	k[2] = h * (R_b + w[2] * (w[0] - R_c));
	//}

	if (model[0] == 'L')
	{
		for (int i = 0; i < n; i++)
		{
			int id = 3 * i;
			double s = 0;

			for (int j = 0; j < n; j++)
				s += CS[j][i] * (w[3 * j] - w[3 * i]);
			
			k[id + 0] = h * (sigma*(w[id + 1] - w[id + 0]) + I_input + s);
			k[id + 1] = h * (rho*w[id + 0] - w[id + 1] - w[id + 0] * w[id + 2]);
			k[id + 2] = h * (-beta * w[id + 2] + w[id + 0] * w[id + 1]);
		}
	}
	else
	{
		for (int i = 0; i < n; i++)
		{
			int id = 3 * i;
			double s = 0;

			for (int j = 0; j < n; j++)
				s += CS[j][i] * (w[3 * j] - w[3 * i]);

			k[id + 0] = h * (-w[id + 1] - w[id + 2] + I_input + s);
			k[id + 1] = h * (w[id + 0] + R_a * w[id + 1]);
			k[id + 2] = h * (R_b + w[id + 2] * (w[id + 0] - R_c));
		}
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


	//double x_start = a.x, dx_start = a.dx;
	//double w[3], w0[3];  // x,y,z
	//double k[4][3];      // x,y,z

	for (int i = 0; i < n; i++)
	{
		x_start[i] = a[i].x;
		dx_start[i] = a[i].dx;
		w[i * 3 + 0] = a[i].x;
		w[i * 3 + 1] = a[i].y;
		w[i * 3 + 2] = a[i].z;
		a[i].if_fired = 0;
	}


	for (int i = 0; i < 3*n; i++)
		w0[i] = w[i];
	Update_once_RK4(k[0], w0, n, t, dt);            //1	

	for (int i = 0; i < 3*n; i++)
		w0[i] = w[i] + 0.5*k[0][i];
	Update_once_RK4(k[1], w0, n, t + dt / 2, dt);   //2


	for (int i = 0; i < 3*n; i++)
		w0[i] = w[i] + 0.5*k[1][i];
	Update_once_RK4(k[2], w0, n, t + dt / 2, dt);   //3


	for (int i = 0; i < 3*n; i++)
		w0[i] = w[i] + k[2][i];
	Update_once_RK4(k[3], w0, n, t + dt, dt);   //4

	for (int i = 0; i < 3*n; i++)
		w[i] = w[i] + (k[0][i] + k[1][i] * 2 + k[2][i] * 2 + k[3][i]) / 6.0;


	for (int i = 0; i < n; i++)
	{
		a[i].x = w[i * 3 + 0];
		a[i].y = w[i * 3 + 1];
		a[i].z = w[i * 3 + 2];
		a[i].t = t + dt;


		if (I_CONST)
			a[i].I_input = I_const_input;
		else
			a[i].I_input = 0;

		double s = 0;

		for (int j = 0; j < n; j++)
			s += CS[j][i] * (a[j].x - a[i].x);
		
		if(model[0] == 'L')
			a[i].dx = sigma * (a[i].y - a[i].x) + a[i].I_input + s;
		else
			a[i].dx = -a[i].y - a[i].z + a[i].I_input + s;

		if (x_start[i] < x_th && a[i].x >= x_th && t + dt - a[i].last_fire_time >= T_ref)
		{
			a[i].last_fire_time = cubic_hermite_real_root(t, t + dt, x_start[i], a[i].x, dx_start[i], a[i].dx, x_th);
			a[i].fire_num++;
			a[i].if_fired = 1;
			if (a[i].x < x_th)
				a[i].x = x_th;

			if (record_data[0])
			{
				fwrite(&a[i].last_fire_time, sizeof(double), 1, FP);
				double s = i;
				fwrite(&s, sizeof(double), 1, FP);
			}
		}

		if (abs(a[i].x) > 1e2)
		{
			printf("\nError! Too large time step %0.6f in RK4\n", T_step);
			printf("id=%d dt=%0.2e last=%f t=%f x=%0.2e dx=%0.2e state=%d\n\n", i, dt, a[i].last_fire_time, a[i].t, a[i].x, a[i].dx, a[i].state);
			getchar();// system("pause");
			exit(1);
		}
		
	}


}

