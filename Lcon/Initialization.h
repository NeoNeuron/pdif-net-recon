double Decide_S(int i, int j, long &seed, long &seed1)  // i-->j
{
	double s;  // scaling
	if (i < NE && j < NE)
		s = S[0];
	else if (i < NE && j >= NE)
		s = S[1];
	else if (i >= NE && j < NE)
		s = S[2];
	else
		s = S[3];
	
	if (random_S == 0)
		return s;
	else if (random_S == 1)
		return Random(seed) * 2 * s;
	else if (random_S == 2)
		return abs(sqrt(-2 * log(Random(seed)))*cos(2 * PI*Random(seed1))*s / 4 + s);
	else if (random_S == 3)
		return -log(1 - Random(seed)) * s;
	else
	{
		double sig, mu;
		sig = 0.794, mu = log(s) - sig * sig / 2;
		double b = sqrt(-2 * log(Random(seed)))*cos(2 * PI*Random(seed1))*sig + mu;
		return exp(b);
	}
}

void Create_connect_matrix(long& seed)
{
	Connect_Matrix = new double *[N];
	for (int i = 0; i < N; i++)
		Connect_Matrix[i] = new double[N];

	CS = new double *[N];
	for (int i = 0; i < N; i++)
		CS[i] = new double[N];
	long seed1 = 15, seed2 = 43; // seed for random S

	for (int i = 0; i < 1000; i++)
	{
		Random(seed1), Random(seed2);
	}


	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			Connect_Matrix[i][j] = 0;
			CS[i][j] = 0;
		}


	if (N == 3)
	{
		Connect_Matrix[0][1] = 1;
		Connect_Matrix[1][2] = 1;

		CS[0][1] = S[0];			
		CS[1][2] = S[0];
	}
	else if( N == 4 || N == 5)
	{
		Connect_Matrix[0][1] = 1;
		Connect_Matrix[1][2] = 1;
		Connect_Matrix[0][3] = 1;

		CS[0][1] = S[0];
		CS[1][2] = S[0];
		CS[0][3] = S[0];
	}
	else
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				if (i != j && Random(seed) < P_c)
				{
					Connect_Matrix[i][j] = 1;
					CS[i][j] = Decide_S(i, j, seed1, seed2);
				}
			}
		}
	}
}

void Assign_CS()
{
	CS = new double *[N];
	for (int i = 0; i < N; i++) {
		CS[i] = new double[N];
		for (int j = 0; j < N; j++) {
			if (Connect_Matrix[i][j] == 1)
				CS[i][j] = S[0];
		}
	}
}

void Record_connect_matrix()
{
	if (record_data[0] || record_data[1] || record_data[2] || record_data[3])
	{
		FILE *fp;
		char str[200], ch[10], c[10];

		strcpy(str, file), strcat(str, "connect_matrix-p=");
		sprintf(ch, "%0.3f", P_c), strcat(str, ch);

		if (random_S == 1)
			strcat(str, "-U");
		else if (random_S == 2)
			strcat(str, "-G");
		else if (random_S == 3)
			strcat(str, "-E");
		else if (random_S == 4)
			strcat(str, "-LN");
		strcat(str, ".dat");
		
		if ((fp = fopen(str, "rb")) == NULL)
		{
			fp = fopen(str, "wb");
			for (int i= 0; i < N; i++)
				fwrite(Connect_Matrix[i], sizeof(double), N, fp);

			if (random_S != 0)
				for (int i = 0; i < N; i++)
					fwrite(CS[i], sizeof(double), N, fp);
			fclose(fp);
		}

	}


	if (record_data[0] || record_data[1] || record_data[2] || record_data[3])
	{
		FILE *fp;
		char str[200], ch[10];

		strcpy(str, file), strcat(str, "connect_matrix-p=");
		sprintf(ch, "%0.3f", P_c), strcat(str, ch);

		if (random_S == 1)
			strcat(str, "-U");
		else if (random_S == 2)
			strcat(str, "-G");
		else if (random_S == 3)
			strcat(str, "-E");
		else if (random_S == 4)
			strcat(str, "-LN");
		strcat(str, ".dat");

		if ((fp = fopen(str, "rb")) == NULL)
		{
			fp = fopen(str, "wb");
			for (int i = 0; i < N; i++)
				fwrite(Connect_Matrix[i], sizeof(double), N, fp);

			if (random_S != 0)
				for (int i = 0; i < N; i++)
					fwrite(CS[i], sizeof(double), N, fp);
			fclose(fp);
		}

	}
}

void Initialization(long &seed0,long &seed2)
{
	S[1] = S[0]; S[2] = S[0]; S[3] = S[0];

	if (model[0] == 'L')
		x_th = L_x_th;
	else
		x_th = R_x_th;

	if (full_toggle) {
		Assign_CS();
	} else {
		Create_connect_matrix(seed0);
	}
	Record_connect_matrix();
	neu = new struct neuron[N];
	neu_old = new struct neuron[N];

	if (TrialID) // multiple trials case for init seeds
	{
		seed2 += TrialID;
	}

	long Seed = 11, Seed1 = 43;
	for (int i = 0; i < 1000; i++)
	{
		Random(Seed), Random(Seed1);
	}

	for (int i = 0; i < N; i++)
	{
		neu[i].t = 0;

		if (model[0] == 'L')
		{
			neu[i].x = (Random(seed2) - 0.5) * 10;
			for (int j = 0; j < 500; j++)
				Random(seed2);
			neu[i].dx = 0;
			neu[i].y = (Random(seed2) - 0.5) * 10;
			for (int j = 0; j < 500; j++)
				Random(seed2);
			neu[i].z = (Random(seed2) - 0.5) * 10 + 15;
			for (int j = 0; j < 500; j++)
				Random(seed2);
		}
		else
		{
			neu[i].x = (Random(seed2) - 0.5) * 10;
			for (int j = 0; j < 500; j++)
				Random(seed2);
			neu[i].dx = 0;
			neu[i].y = Random(seed2) * 7 - 5;
			for (int j = 0; j < 500; j++)
				Random(seed2);
			neu[i].z = 0;
			for (int j = 0; j < 500; j++)
				Random(seed2);		
		}

		neu[i].I_input = 0;
		neu[i].fire_num = 0;
		neu[i].last_fire_time = -1e5;
		neu[i].if_fired = 0;
		for (int j = 0; j < 500; j++)
			Random(seed2);
		neu[i].seed = seed2;

		neu[i].wait_strength_E = 0;
		neu[i].wait_strength_I = 0;

		neu[i].state = 1;
		neu_old[i].state = 0;
	}


	x_start = new double[N];
	dx_start = new double[N];
	w = new double[3 * N];
	w0 = new double[3 * N];

	k = new double *[4];
	for (int i = 0; i < 4; i++)
		k[i] = new double[3 * N];

	
}
