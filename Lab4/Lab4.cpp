#include <mpi.h>
#include <iostream>
#include <vector>

int ProcNum;
int ProcRank;

void generateMatrix(double* matrix, int Size) {
	for (int i = 0; i < Size; ++i) {
		for (int j = 0; j < Size; ++j) {
			matrix[i * Size + j] = rand() % 10;
		}
	}
}

void printMatrix(const double* matrix, int Size) {
	for (int i = 0; i < Size; ++i) {
		for (int j = 0; j < Size; ++j) {
			std::cout << matrix[i * Size + j] << '\t';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void Flip(double* A, int Size) {
	for (int i = 0; i < Size; ++i) {
		for (int j = i + 1; j < Size; ++j) {
			double temp = A[i * Size + j];
			A[i * Size + j] = A[j * Size + i];
			A[j * Size + i] = temp;
		}
	}
}

void MatrixMultiplicationMPI(double*& A, double*& B, double*& C, int& Size) {
	int dim = Size;
	int i, j, k, p, ind;
	double temp;
	MPI_Status Status;
	int ProcPartSize = dim / ProcNum;
	int ProcPartElem = ProcPartSize * dim;
	double* bufA = new double[ProcPartElem];
	double* bufB = new double[ProcPartElem];
	double* bufC = new double[ProcPartElem];
	int ProcPart = dim / ProcNum, part = ProcPart * dim;
	if (ProcRank == 0) {
		Flip(B, Size);
	}

	MPI_Scatter(A, part, MPI_DOUBLE, bufA, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(B, part, MPI_DOUBLE, bufB, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	temp = 0.0;
	for (i = 0; i < ProcPartSize; i++) {
		for (j = 0; j < ProcPartSize; j++) {
			for (k = 0; k < dim; k++)
				temp += bufA[i * dim + k] * bufB[j * dim + k];
			bufC[i * dim + j + ProcPartSize * ProcRank] = temp;
			temp = 0.0;
		}
	}

	int NextProc; int PrevProc;
	for (p = 1; p < ProcNum; p++) {
		NextProc = ProcRank + 1;
		if (ProcRank == ProcNum - 1)
			NextProc = 0;
		PrevProc = ProcRank - 1;
		if (ProcRank == 0)
			PrevProc = ProcNum - 1;
		MPI_Sendrecv_replace(bufB, part, MPI_DOUBLE, NextProc, 0, PrevProc, 0, MPI_COMM_WORLD, &Status);
		temp = 0.0;
		for (i = 0; i < ProcPartSize; i++) {
			for (j = 0; j < ProcPartSize; j++) {
				for (k = 0; k < dim; k++) {
					temp += bufA[i * dim + k] * bufB[j * dim + k];
				}
				if (ProcRank - p >= 0)
					ind = ProcRank - p;
				else ind = (ProcNum - p + ProcRank);
				bufC[i * dim + j + ind * ProcPartSize] = temp;
				temp = 0.0;
			}
		}
	}

	MPI_Gather(bufC, ProcPartElem, MPI_DOUBLE, C, ProcPartElem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	delete[] bufA;
	delete[] bufB;
	delete[] bufC;
}

int main(int argc, char* argv[]) {
	int Size = 6;

	double* A = new double[Size * Size];
	double* B = new double[Size * Size];
	double* C = new double[Size * Size];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

	
	if (ProcRank == 0) {
		generateMatrix(A, Size);
		printMatrix(A, Size);
		
		generateMatrix(B, Size);
		printMatrix(B, Size);
	}

	MatrixMultiplicationMPI(A, B, C, Size);

	MPI_Finalize();

	if (ProcRank == 0) {
		printMatrix(C, Size);
	}

	return 0;
}
