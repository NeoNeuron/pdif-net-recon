void Delete()
{
	for (int i = 0; i < N; i++)
	{
		delete[] Connect_Matrix[i];
		delete[] CS[i];
	}
	delete[] Connect_Matrix;
	delete[] CS;
	delete[] neu, delete[] neu_old;

	if (record_data[0])
		fclose(FP); 
	if (record_data[1])
		fclose(FPx);
	if (record_data[2])
		fclose(FPy);
	if (record_data[3])
		fclose(FPz);
	
	delete[] x_start;
	delete[] dx_start;
	delete[] w;
	delete[] w0;
	for (int i = 0; i < 4; i++)
		delete[] k[i];
	delete[] k;


}