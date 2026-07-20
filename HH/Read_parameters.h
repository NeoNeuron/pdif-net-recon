#include "mkdir.h"
#include "npy_io.h"

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
	if (full_toggle < 0 || full_toggle > 2)
	{
		fprintf(stderr, "Error in Read_parameters()! full_mode=%d, must be 0 (random), 1 (full CLI list via --conn_matrix), or 2 (load from --conn_matrix_file).\n", full_toggle);
		exit(1);
	}
    overwrite_conn = vm["overwrite_conn"].as<int>();
    strcpy(fi_conn_matrix, vm["conn_matrix_file"].as<string>().c_str());

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
	if (full_toggle == 1) {
        vector<double> conn_buff;
        str2vec(vm["conn_matrix"].as<string>(), conn_buff);
		// Create the read the connect_matrix
		Connect_Matrix = new double *[N];
		for (int i = 0; i < N; i++) {
			Connect_Matrix[i] = new double[N];
			for (int j = 0; j < N; j++)
				Connect_Matrix[i][j] = conn_buff[i*N+j];
		}
	} else if (full_toggle == 2) {
		if (fi_conn_matrix[0] == '\0') {
			fprintf(stderr, "Error in Read_parameters()! full_mode=2 requires --conn_matrix_file to be set.\n");
			exit(1);
		}
		vector<size_t> shape;
		vector<double> conn_buff = load_npy<double>(fi_conn_matrix, shape);
		if (shape.size() != 2 || (int)shape[0] != N || (int)shape[1] != N) {
			fprintf(stderr, "Error in Read_parameters()! conn_matrix_file '%s' has shape (", fi_conn_matrix);
			for (size_t s : shape)
				fprintf(stderr, "%zu ", s);
			fprintf(stderr, "), expected (%d, %d).\n", N, N);
			exit(1);
		}

		Connect_Matrix = new double *[N];
		CS = new double *[N];
		for (int i = 0; i < N; i++) {
			Connect_Matrix[i] = new double[N];
			CS[i] = new double[N];
			for (int j = 0; j < N; j++) {
				double w = conn_buff[i*N+j];
				if (w < 0) {
					fprintf(stderr, "Error in Read_parameters()! conn_matrix_file '%s' has negative weight %.6g at (%d, %d). CS must be non-negative; E/I type comes from the presynaptic index, not the weight's sign.\n", fi_conn_matrix, w, i, j);
					exit(1);
				}
				Connect_Matrix[i][j] = (w != 0) ? 1 : 0;
				CS[i][j] = w;
			}
		}

		if (random_S != 0) {
			printf("NOTE: full_mode=2 loads CS directly from conn_matrix_file; random_S=%d is ignored.\n", random_S);
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

	CP = vm["CP"].as<double>();
    Lyapunov = vm["Lyapunov"].as<int>();
    record_data[0] = vm["record_spk"].as<int>();
    record_data[1] = vm["record_v"].as<int>();

    vector<double> vlim_buff;
    str2vec(vm["record_vlim"].as<string>(), vlim_buff);
    Record_v_start = vlim_buff[0];
    Record_v_end = vlim_buff[1];

    strcpy(file, vm["record_path"].as<string>().c_str());
    strcpy(fi_neu_state, vm["state_path"].as<string>().c_str());
    strcpy(save_mode, vm["save_mode"].as<string>().c_str());

	// if no neu_state_file
	if ((fi_neu_state == NULL || fi_neu_state[0] == '\0') && strcmp(save_mode, "w")) {
		WARNING("WARNING: no fi_neu_state provided! Force save_mode to 'w'.\n");
		strcpy(save_mode, "w");
    }

	// deprecated: remove the creation of subfolders according the network size and type
	// if (N == NE)
	// 	strcat(file, "EE/N=");
	// else if (N == NI)
	// 	strcat(file, "II/N=");
	// else
	// 	strcat(file, "EI/N=");
	// sprintf(ch, "%d", N), strcat(file, ch), strcat(file, "/");

	TrialID = vm["TrialID"].as<int>();
	// initialize folder
	make_dir_recursive(file);
}


void out_put_filename()
{
	ode_type = 2;
	IntOrd = 1;
	method = 3;
	// T_step = 0.2;
	Power_spectrum = 0;
	Estimate_RK4_call = 0;
	RecordFP = 0;
	if (fabs(T_step - int(T_step)) < 1e-8)
		T_step = pow(0.5, T_step);
	
	S_d_e = exp(-T_step / Sigma_d_E);
	S_d_i = exp(-T_step / Sigma_d_I);
	S_r_e = exp(-T_step / Sigma_r_E);
	S_r_i = exp(-T_step / Sigma_r_I);

	Regular_method = 0, Lib_method = 0, Adaptive_method = 0, ETDRK_method = 0, ETD_method = 0;

	if (method <= 1)
		Regular_method = 1;
	else if (method == 2)
		Lib_method = 1;
	else if (method == 3)
		ETDRK_method = 1;
	else if (method == 4)
		ETD_method = 1;

	char str[200] = "", c[10], str1[200];

	if (Lyapunov)
	{
		record_data[0] = 0;
		record_data[1] = 0;
	}

	if (CP < 1e-6)
		strcpy(str, "HH");
	else
	{
		strcpy(str, "P"), sprintf(c, "%0.2f", CP), strcat(str, c);
		strcat(str, "HH");
	}

	if (random_S == 1)
		strcat(str, "U-");
	else if (random_S == 2)
		strcat(str, "G-");
	else if (random_S == 3)
		strcat(str, "E-");
	else if (random_S == 4)
		strcat(str, "LN-");

	strcat(str, "p="), sprintf(c, "%0.2f", P_c), strcat(str, c);
	if (NE) {
		if (S[0]<1e-3 && S[0]>1e-10)
			strcat(str, "s="), sprintf(c, "%0.5f", S[0]), strcat(str, c);  
		else
			strcat(str, "s="), sprintf(c, "%0.3f", S[0]), strcat(str, c);
	} else {
		if (S[2]<1e-3 && S[2]>1e-10)
			strcat(str, "s="), sprintf(c, "%0.5f", S[2]), strcat(str, c);  
		else
			strcat(str, "s="), sprintf(c, "%0.3f", S[2]), strcat(str, c);
	}
	if (NE && NI) {
		if (S[2]<1e-3 && S[2]>1e-10)
			strcat(str, "s="), sprintf(c, "%0.5f", S[2]), strcat(str, c);  
		else
			strcat(str, "s="), sprintf(c, "%0.3f", S[2]), strcat(str, c);
	}
	strcat(str, "f="), sprintf(c, "%0.3f", f[0]), strcat(str, c);
	strcat(str, "u="), sprintf(c, "%0.3f", Nu), strcat(str, c);
	

	if (random_Nu == 1)
		strcat(str, "-NuU");
	else if (random_Nu == 2)
		strcat(str, "-NuG");
	else if (random_Nu == 3)
		strcat(str, "-NuE");
	else if (random_Nu == 4)
		strcat(str, "-NuLN");

	strcpy(filename, str);
	printf("T_Max=%0.2e dt=%0.3f\n", T_Max,T_step);

	if (TrialID)  // for multiple trials with fixed CS and Poisson seeds 
	{
		strcat(str, "-"), sprintf(c, "%d", TrialID), strcat(str, c);
	}

	// copy path to load neuron states
	if (fi_neu_state != NULL && fi_neu_state[0] != '\0') {
		char str_buff[200];
		strcpy(str_buff, file), strcat(str_buff, fi_neu_state);
		strcpy(fi_neu_state, str_buff);
	}
	// copy path to save neuron states
	strcpy(str1, file), strcat(str1, str), strcat(str1, "_state.dat");
	strcpy(fo_neu_state, str1);

	// create FILE object to save spike train and voltage
    char open_mode[3];
    strcpy(open_mode, save_mode), strcat(open_mode, "b");
	if (record_data[0])
	{
		strcpy(str1, file), strcat(str1, str), strcat(str1, "_spike_train.dat");
		FP = fopen(str1, open_mode);
	}
	if (record_data[1])
	{
		strcpy(str1, file), strcat(str1, str), strcat(str1, "_voltage.dat");
		FP1 = fopen(str1, open_mode);
	}

	if (Power_spectrum)				
	{
		char ch[200];
		strcpy(ch, file), strcat(ch, "fftw_"), strcat(ch, str), strcat(ch, ".dat");
		FP_FFTW = fopen(ch, "wb");
	}

	if (RecordFP)
	{
		if (N < 100)
		{
			printf("N=%d should >= 100 to record FP!\n", N);
			exit(0);
		}
		char ch[200];
		strcpy(ch, file), strcat(ch, "FP_"), strcat(ch, str), strcat(ch, ".dat");
		FP_fire_pattern = fopen(ch, "wb");
	}

	if (record_data[0] || record_data[1] || Power_spectrum || RecordFP)
	{
		if (NE == N)
			printf("file:NE=%d    %s\n", N, str);
		else if (NI == N)
			printf("file:NI=%d    %s\n", N, str);
		else
			printf("file:NEI=%d    %s\n", N, str);
	}
	
}