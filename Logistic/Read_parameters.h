#include "mkdir.h"

void Read_parameters(po::variables_map& vm)
{
	char ch[100];
	NE = vm["NE"].as<int>();
	NI = vm["NI"].as<int>();
	N = NE + NI;
	T_Max = vm["T_Max"].as<double>();
	T_step = vm["T_step"].as<double>();

    vector<double> s_buff;
	str2vec(vm["S"].as<string>(), s_buff);
	for (int i=0; i<4; i++)
		S[i] = s_buff[i];

    I_CONST = vm["I_CONST"].as<double>();

	// full-version config toggle:
    full_toggle = vm["full_mode"].as<int>();
	// CS
	if (full_toggle) {
        vector<double> conn_buff;
        str2vec(vm["conn_matrix"].as<string>(), conn_buff);
		// Create the read the connect_matrix
		Connect_Matrix = new double *[N];
		for (int i = 0; i < N; i++) {
			Connect_Matrix[i] = new double[N];
			for (int j = 0; j < N; j++)
				Connect_Matrix[i][j] = conn_buff[i*N+j];
		}
	}

    P_c = vm["P_c"].as<double>();
    random_S = vm["random_S"].as<int>();
	if (random_S > 4 || random_S < 0)
	{
		printf("Error random_S=%d\n", random_S);
		getchar();
		exit(0);
	}

    Lyapunov = vm["Lyapunov"].as<int>();
    record_data[0] = vm["record_spk"].as<int>();
    record_data[1] = vm["record_v"].as<int>();

    vector<double> xlim_buff;
    str2vec(vm["record_vlim"].as<string>(), xlim_buff);
    Record_x_start = xlim_buff[0];
    Record_x_end   = xlim_buff[1];

    strcpy(file, vm["record_path"].as<string>().c_str());

	if (N == NE)
		strcat(file, "EE/N=");
	else if (N == NI)
		strcat(file, "II/N=");
	else
		strcat(file, "EI/N=");
	sprintf(ch, "%d", N), strcat(file, ch), strcat(file, "/");

	// initialize folder
	_mkdir(file);
}


void out_put_filename()
{
	// T_step = 1;

	char str[200] = "", c[10], str1[200];

	if (Lyapunov)
	{
		record_data[0] = 0;
		record_data[1] = 0;
	}

	strcpy(str, "Log");

	if (random_S == 1)
		strcat(str, "U-");
	else if (random_S == 2)
		strcat(str, "G-");
	else if (random_S == 3)
		strcat(str, "E-");
	else if (random_S == 4)
		strcat(str, "LN-");

	/*strcat(str, "RK4_"), strcat(str, "t="), sprintf(c, "%0.2f", T_step), strcat(str, c);*/

	strcat(str, "p="), sprintf(c, "%0.2f", P_c), strcat(str, c);
	if (S[0]<1e-3 && S[0]>1e-10)
		strcat(str, "s="), sprintf(c, "%0.5f", S[0]), strcat(str, c);  
	else
		strcat(str, "s="), sprintf(c, "%0.3f", S[0]), strcat(str, c);
	if (NE && NI)
	{
		if (S[2]<1e-3 && S[2]>1e-10)
			strcat(str, "s="), sprintf(c, "%0.5f", S[2]), strcat(str, c);  
		else
			strcat(str, "s="), sprintf(c, "%0.3f", S[2]), strcat(str, c);
	}

	printf("dt=%0.3f, T_Max=%0.2e\n", T_step, T_Max);

	if (record_data[0])
	{
		strcpy(str1, file), strcat(str1, str), strcat(str1, "_spike_train.dat");
		FP = fopen(str1, "wb");
	}
	if (record_data[1])
	{
		strcpy(str1, file), strcat(str1, str), strcat(str1, "_voltage.dat");
		FP1 = fopen(str1, "wb");
	}

	if (record_data[0] || record_data[1])
	{
		if (NE == N)
			printf("file:NE=%d\\%s\n", N, str);
		else if (NI == N)
			printf("file:NI=%d\\%s\n", N, str);
		else
			printf("file:NEI=%d\\%s\n", N, str);
	}
	
}