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

	// Nu
    Nu = vm["Nu"].as<double>();
	// full-version config toggle:
    full_toggle = vm["full_mode"].as<int>();

	// f
	f = new double[N]{0};
	if (full_toggle) { // load f for all neurons.
        vector<double> f_buff;
        str2vec(vm["f"].as<string>(), f_buff);
		for (int i=0; i<N; i++) {
            f[i] = f_buff[i];
            printf("%.2f\t", f[i]);
		}
	} else { // load f for E and I types, for each type of neuron, f is identical.
        fE = vm["fE"].as<double>();
        fI = vm["fI"].as<double>();
        for (int i=0; i<NE; i++)
            f[i] = fE;
        for (int i=0; i<NI; i++)
            f[i+NE] = fI;
        printf("f = (E : %.3lf, I : %.3lf)\n", fE, fI);
	}

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
    random_Nu = vm["random_Nu"].as<int>();
	if (random_S > 4 || random_S < 0)
	{
		printf("Error random_S=%d\n", random_S);
		getchar();
		exit(0);
	}

	if (random_Nu > 4 || random_Nu < 0)
	{
		printf("Error random_Nu=%d\n", random_Nu);
		getchar();
		exit(0);
	}


    Lyapunov = vm["Lyapunov"].as<int>();
    record_data[0] = vm["record_spk"].as<int>();
    record_data[1] = vm["record_v"].as<int>();

    vector<double> xlim_buff;
    str2vec(vm["record_vlim"].as<string>(), xlim_buff);
    Record_x_start = xlim_buff[0];
    Record_x_end = xlim_buff[1];

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
	char str[200] = "", c[10], str1[200];

	if (Lyapunov)
	{
		record_data[0] = 0;
		record_data[1] = 0;
	}

	strcpy(str, "L");

	if (random_S == 1)
		strcat(str, "U-");
	else if (random_S == 2)
		strcat(str, "G-");
	else if (random_S == 3)
		strcat(str, "E-");
	else if (random_S == 4)
		strcat(str, "LN-");

	strcat(str, "p="), sprintf(c, "%0.2f", P_c), strcat(str, c);
	strcat(str, "s="), sprintf(c, "%0.3f", S[0]), strcat(str, c);
	if (NE && NI)
	{
		strcat(str, "s="), sprintf(c, "%0.3f", S[2]), strcat(str, c);
	}
	strcat(str, "f="), sprintf(c, "%0.3f", f[0]), strcat(str, c);
	strcat(str, "u="), sprintf(c, "%0.3f", Nu), strcat(str, c);


	if (random_Nu == 1)
		strcat(str, "U");
	else if (random_Nu == 2)
		strcat(str, "G");
	else if (random_Nu == 3)
		strcat(str, "E");
	else if (random_Nu == 4)
		strcat(str, "LN");

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