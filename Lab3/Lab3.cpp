#include <iostream>
#include <vector>
#include "omp.h"

class NoSolutionError : public std::exception {
public:
    const char* what() const noexcept override {
        return "Отсутствует решение";
    }
};

// Последовательные функции
template <typename T>
int col_max(const std::vector<std::vector<T>>& matrix, int col, int n) {
    T max = std::abs(matrix[col][col]);
    int maxPos = col;
    for (int i = col + 1; i < n; ++i) {
        T element = std::abs(matrix[i][col]);
        if (element > max) {
            max = element;
            maxPos = i;
        }
    }
    return maxPos;
}

template <typename T>
int triangulation(std::vector<std::vector<T>>& matrix, int n) {
    unsigned int swapCount = 0;
    if (0 == n)
        return swapCount;
    const int num_cols = matrix[0].size();
    for (int i = 0; i < n - 1; ++i) {
        unsigned int imax = col_max(matrix, i, n);
        if (i != imax) {
            swap(matrix[i], matrix[imax]);
            ++swapCount;
        }
        for (int j = i + 1; j < n; ++j) {
            T mul = -matrix[j][i] / matrix[i][i];
            for (int k = i; k < num_cols; ++k) {
                matrix[j][k] += matrix[i][k] * mul;
            }
        }
    }
    return swapCount;
}

// Параллельные функции
template <typename T>
int col_max_parallel(const std::vector<std::vector<T>>& matrix, int col, int n) {
    T max = std::abs(matrix[col][col]);
    int maxPos = col;
#pragma omp parallel
    {
        T loc_max = max;
        T loc_max_pos = maxPos;
    #pragma omp for
        for (int i = col + 1; i < n; ++i) {
            T element = std::abs(matrix[i][col]);
            if (element > loc_max) {
                loc_max = element;
                loc_max_pos = i;
            }
        }
    #pragma omp critical
        {
            if (max < loc_max) {
                max = loc_max;
                maxPos = loc_max_pos;
            }
        }
    }
    return maxPos;
}

template <typename T>
int triangulation_parallel(std::vector<std::vector<T>>& matrix, int n) {
    unsigned int swapCount = 0;
    if (0 == n)
        return swapCount;
    const int num_cols = matrix[0].size();
    for (int i = 0; i < n - 1; ++i) {
        unsigned int imax = col_max_parallel(matrix, i, n);
        if (i != imax) {
            swap(matrix[i], matrix[imax]);
            ++swapCount;
        }
#pragma omp parallel for
        for (int j = i + 1; j < n; ++j) {
            T mul = -matrix[j][i] / matrix[i][i];
            for (int k = i; k < num_cols; ++k) {
                matrix[j][k] += matrix[i][k] * mul;
            }
        }
    }
    return swapCount;
}

// Решение
template <typename T>
std::vector<T> gauss_solving(std::vector<std::vector<T>>& matrix,
    std::vector<T>& free_term_column, int n) {
    std::vector<T> solution(n);
    for (int i = 0; i < n; ++i) {
        matrix[i].push_back(free_term_column[i]);
    }
    triangulation_parallel(matrix, n);
    for (int i = n - 1; i >= 0; --i) {
        if (std::abs(matrix[i][i]) < 0.0001)
            throw NoSolutionError();
        solution[i] = matrix[i][n] / matrix[i][i];
        for (int j = 0; j < i; ++j) {
            matrix[j][n] -= matrix[j][i] * solution[i];
        }
    }
    return solution;
}


int main() {
    for (int n = 500; n <= 3000; n += 500) {
        std::vector<std::vector<double>> matrix(n);
        for (int i = 0; i < n; ++i) {
            matrix[i].resize(n);
            for (int j = 0; j < n; ++j)
                matrix[i][j] = rand();
        }
        std::vector<double> column(n);
        for (int j = 0; j < n; ++j)
            column[j] = rand();
        double start_time = omp_get_wtime();
        std::vector<double> solution = gauss_solving(matrix, column, n);
        std::cout << n << " " << omp_get_wtime() - start_time << std::endl;
    }
}