#include "iostream"
#include "mpi.h"
#include "cassert"
using namespace std;


int main(int argc, char** argv)
{
	double elapsed_time;
	int i, j, k, index;
	int ProcRank, ProcNum;
	double scale;
	int size;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	if (ProcRank == 0)
	{
		cout << "Number of processors: " << ProcNum << endl;
		cout << "Input size: ";
		cin >> size;
		assert(size % ProcNum == 0);
		for (int i = 1; i < ProcNum; i++)
			MPI_Send(&size, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
	}
	if (ProcRank != 0)
		MPI_Recv(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

	elapsed_time = MPI_Wtime();
	double** ttA = new double* [size];
	for (int l = 0; l < size; l++)
		ttA[l] = new double[size + 1];
	for (int l = 0; l < size; l++)
	{
		for (int m = 0; m < size + 1; m++)

			ttA[l][m] = 9 * (int)rand() / RAND_MAX + 1;


	}
	/*if (!ProcRank)
		for (int l = 0; l < size; l++)
		{
			for (int m = 0; m < size + 1; m++)
			{

				cout << ttA[l][m] << " ";
			}
			cout << endl;
		}*/

	int numrows = size / ProcNum;

	double** A_Local = new double* [numrows];
	for (i = 0; i < numrows; i++)
		A_Local[i] = new double[size + 1];

	int* myrows = new int[numrows];


	for (i = 0; i < numrows; i++)
	{
		index = ProcRank + i * ProcNum;
		myrows[i] = index;
		for (int j = 0; j < size + 1; j++)
			A_Local[i][j] = ttA[index][j];
	}


	double* pivot = new double[size + 1];
	


	int cnt = 0;
	for (i = 0; i < size - 1; i++)
	{
		if (i == myrows[cnt])
		{
			MPI_Bcast(A_Local[cnt], size + 1, MPI_DOUBLE, ProcRank, MPI_COMM_WORLD);
			for (j = 0; j < size + 1; j++)
				pivot[j] = A_Local[cnt][j];
			cnt++;
		}
		else
			MPI_Bcast(pivot, size + 1, MPI_DOUBLE, i % ProcNum, MPI_COMM_WORLD);

		for (j = cnt; j < numrows; j++)
		{
			scale = A_Local[j][i] / pivot[i];
			for (k = i; k < size + 1; k++)
				A_Local[j][k] = A_Local[j][k] - scale * pivot[k];
		}
	}


	double* x = new double[size];

	cnt = 0;
	for (i = 0; i < size; i++)
	{
		if (i == myrows[cnt])
		{
			x[i] = A_Local[cnt][size];
			cnt++;
		}
		else
			x[i] = 0;
	}

	cnt = numrows - 1;
	for (i = size - 1; i > 0; i--)
	{
		if (cnt >= 0)
		{
			if (i == myrows[cnt])
			{
				x[i] = x[i] / A_Local[cnt][i];
				MPI_Bcast(x + i, 1, MPI_DOUBLE, ProcRank, MPI_COMM_WORLD);
				cnt--;
			}
			else
				MPI_Bcast(x + i, 1, MPI_DOUBLE, i % ProcNum, MPI_COMM_WORLD);
		}
		else
			MPI_Bcast(x + i, 1, MPI_DOUBLE, i % ProcNum, MPI_COMM_WORLD);


		for (j = 0; j <= cnt; j++)
			x[myrows[j]] = x[myrows[j]] - A_Local[j][i] * x[i];
	}


	if (ProcRank == 0)
	{
		x[0] = x[0] / A_Local[cnt][0];
		MPI_Bcast(x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	else
		MPI_Bcast(x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	elapsed_time = MPI_Wtime() - elapsed_time;
	if (ProcRank == 0)
	{
		/*for (i = 0; i < size; i++)
			cout << "x[" << i << "]=" << x[i] << " ";
		cout << endl;*/
		cout << "Elapsed time: " << elapsed_time << " sec" << endl;
	}

	delete[] pivot;
	delete[] myrows;
	for (i = 0; i < numrows; i++)
		delete[] A_Local[i];
	delete[] A_Local;
	for (i = 0; i < size; i++)
		delete[] ttA[i];
	delete[] ttA;
	MPI_Finalize();

	return 0;
}