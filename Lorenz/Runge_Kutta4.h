//-----------------------------------------------------------------------------
//       G_sse, G_ssi & G_ff update analytically
//-----------------------------------------------------------------------------

void Update_once_RK4(double *k, double *w, int n, double t, double h)
{
	double I_input;
	if (I_CONST)
		I_input = I_const_input;
	else
		I_input = 0;


	k[0] = h * (sigma*(w[1] - w[0]) + I_input);
	k[1] = h * (rho*w[0] - w[1] - w[0] * w[2]);
	k[2] = h * (-beta * w[2] + w[0] * w[1]);

}

//-----------------------------------------------------------------------------
//		Runge Kutta_4 mehtod: update neuron during determinsteristic parts(i.e. no outside input S and F)
//  	start from time t, end with time t+dt. There are four times to update,
//		so can be simplyfied by one function.
//
//		We just update v,m,h,n with RK4 method, since conductance parts are correct solutions.
//-----------------------------------------------------------------------------

void Update_RK4(int n, struct neuron &a, double t, double dt)
{
	double x_start = a.x, dx_start = a.dx;

	double w[3], w0[3];  // x,y,z
	double k[4][3];      // x,y,z


	w[0] = a.x, w[1] = a.y, w[2] = a.z;


	for (int i = 0; i < 3; i++)
		w0[i] = w[i];
	Update_once_RK4(k[0], w0, n, t, dt);            //1	

	for (int i = 0; i < 3; i++)
		w0[i] = w[i] + 0.5*k[0][i];
	Update_once_RK4(k[1], w0, n, t + dt / 2, dt);   //2


	for (int i = 0; i < 3; i++)
		w0[i] = w[i] + 0.5*k[1][i];
	Update_once_RK4(k[2], w0, n, t + dt / 2, dt);   //3


	for (int i = 0; i < 3; i++)
		w0[i] = w[i] + k[2][i];
	Update_once_RK4(k[3], w0, n, t + dt, dt);   //4

	for (int i = 0; i < 3; i++)
		w[i] = w[i] + (k[0][i] + k[1][i] * 2 + k[2][i] * 2 + k[3][i]) / 6.0;


	a.x = w[0], a.y = w[1], a.z = w[2];
	a.t = t + dt;

	if (I_CONST)
		a.I_input = I_const_input;
	else
		a.I_input = 0;

	a.dx = sigma * (a.y - a.x) + a.I_input;

	if (x_start < x_th && a.x >= x_th && t + dt - a.last_fire_time >= T_ref)
	{
		a.last_fire_time = cubic_hermite_real_root(t, t + dt, x_start, a.x, dx_start, a.dx, x_th);
		a.if_fired = 1;
	}

	if (abs(a.x) > 1e2)
	{
		printf("\nError! Too large time step %0.6f in RK4\n", T_step);
		printf("n=%d dt=%0.2e last=%f t=%f x=%0.2e dx=%0.2e state=%d\n\n", n, dt, a.last_fire_time, a.t, a.x, a.dx, a.state);
		getchar();// system("pause");
		exit(1);
	}

}

