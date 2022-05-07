#include "mkdir.h"

void Read_parameters(long &seed, long &seed1)
{
	FILE *fp;
	fp = fopen("NetModel_parameters.txt", "r");

	if (fp == NULL)
		fp = fopen("./Lcon/NetModel_parameters.txt", "r");

	if (fp == NULL)
	{
		printf("Error in Read_parameters()! :: Cann't open parameters input file! \n");
		getchar();// system("pause");
		exit(1);
	}

	if(fp == NULL)
	{
		printf("Error ! Cann't read parameters\n");
		getchar();// system("pause");
		exit(1);
	}

	char ch[100];

	//fscanf(fp, "%s%s", ch, model);
	//while (fgetc(fp) != '\n');
	//if (model[0] == 'L')
	//	strcpy(model, "L");
	//else if(model[0] == 'R')
	//	strcpy(model, "R");
	//else
	//{
	//	printf("Wrong model type %s\n",model);
	//	exit(0);
	//}	

	strcpy(model, "L");

	fscanf(fp,"%s%d%s%d",ch,&NE,ch,&NI);
	N = NE+NI;	
	fscanf(fp, "%s%ld%ld", ch, &seed, &seed1);
	fscanf(fp,"%s%lf%s%lf",ch,&T_Max,ch,&T_step); 
	fscanf(fp, "%s%lf%lf%lf%lf", ch, &S[0], &S[1], &S[2], &S[3]); 
	fscanf(fp, "%s%d", ch, &I_CONST);

	fscanf(fp, "%s%lf%s%lf", ch, &Nu, ch, &f);
	// full-version config toggle:
	fscanf(fp, "%s%d", ch, &full_toggle);

	fscanf(fp, "%s", ch);
	if (full_toggle) {
		// Create the read the connect_matrix
		Connect_Matrix = new double *[N];
		for (int i = 0; i < N; i++) {
			Connect_Matrix[i] = new double[N];
			for (int j = 0; j < N; j++)
				fscanf(fp, "%lf", &Connect_Matrix[i][j]);
		}
	} else { // Connect_Matrix is randomly generated following specific distribution.
		while (fgetc(fp) != '\n');
	}

	fscanf(fp,"%s%lf",ch, &P_c);
	fscanf(fp, "%s%d", ch, &random_S);
	if (random_S > 4 || random_S < 0)
	{
		printf("Error pm.pS=%d\n", random_S);
		getchar();
		exit(0);
	}
	while (fgetc(fp) != '\n');


	fscanf(fp, "%s%d", ch, &random_Nu); 
	if (random_Nu > 4 || random_Nu < 0)
	{
		printf("Error random_Nu=%d\n", random_Nu);
		getchar();
		exit(0);
	}
	while (fgetc(fp) != '\n');

	fscanf(fp, "%s%d", ch, &Lyapunov); 
	fscanf(fp, "%s%d%d",ch, &record_data[0], &record_data[1]);
	fscanf(fp, "%s%lf%lf", ch, &Record_x_start, &Record_x_end);
	fscanf(fp,"%s%s",ch,file);  
	fclose(fp);

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
	//T_step = 0.1;
	char str[200] = "", c[10], str1[200];

	if (Lyapunov)
	{
		record_data[0] = 0;
		record_data[1] = 0;
	}

	if (model[1] != 'c')
		strcat(model, "con");
	strcpy(str, model);

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
	strcat(str, "s="), sprintf(c, "%0.3f", S[0]), strcat(str, c);
	strcat(str, "f="), sprintf(c, "%0.2f", f), strcat(str, c);
	strcat(str, "u="), sprintf(c, "%0.2f", Nu), strcat(str, c);

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