#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

double exec_time[11];
int num_threads;

vector<vector<double>> readMatrixFromFile(const char* filename);
void writeMatrixToFile(const char* filename, const vector<vector<double>>& result);
vector<vector<double>> serial_transpose(const vector<vector<double>>& matrix);
vector<vector<double>> parallel_transpose(const vector<vector<double>>& matrix);
vector<vector<double>> serial_multiplication(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB);
vector<vector<double>> parallel_multiplication(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB);
vector<vector<double>> parallel_multiplication_op(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB);
vector<vector<double>> serial_addition(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB);
vector<vector<double>> parallel_addition(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB);
vector<vector<double>> serial_subtraction(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB);
vector<vector<double>> parallel_subtraction(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB);
vector<vector<double>> serial_inverse(vector<vector<double>>& matrix);
vector<vector<double>> parallel_inverse(vector<vector<double>>& matrix);
vector<vector<double>> complexTransformation(const vector<vector<double>>& matrix);

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <num_threads>" << std::endl;
        return 1;
    }

    const char* inputFileName1 = "gen_matrixA.txt";
    const char* inputFileName2 = "gen_matrixB.txt";
    const char* outputFileName1 = "output_matrix_s_mul.txt";
    const char* outputFileName2 = "output_matrix_p_mul.txt";
    const char* outputFileName3 = "output_matrix_s_trans.txt";
    const char* outputFileName10 = "output_matrix_p_trans.txt";
    const char* outputFileName4 = "output_matrix_s_add.txt";
    const char* outputFileName5 = "output_matrix_p_add.txt";
    const char* outputFileName6 = "output_matrix_s_sub.txt";
    const char* outputFileName7 = "output_matrix_p_sub.txt";
    const char* outputFileName8 = "output_matrix_s_inv.txt";
    const char* outputFileName9 = "output_matrix_p_inv.txt";

    int num_threads = stoi(argv[1]);
    omp_set_num_threads(num_threads);

    vector<vector<double>> inputMatrixA = readMatrixFromFile(inputFileName1);
    vector<vector<double>> inputMatrixB = readMatrixFromFile(inputFileName2);

    if (inputMatrixA.empty() || inputMatrixB.empty()) {
        cerr << "Error: Empty matrix or file not found." << endl;
        return 1;
    }

    vector<vector<double>> result = parallel_multiplication(inputMatrixA,inputMatrixB);
    writeMatrixToFile(outputFileName2, result);
    result = parallel_multiplication_op(inputMatrixA,inputMatrixB);
    result = serial_multiplication(inputMatrixA,inputMatrixB);
    writeMatrixToFile(outputFileName1, result);

    result = serial_transpose(inputMatrixA);
    writeMatrixToFile(outputFileName3, result);
    result = parallel_transpose(inputMatrixA);
    writeMatrixToFile(outputFileName10, result);

    result = serial_addition(inputMatrixA,inputMatrixB);
    writeMatrixToFile(outputFileName4, result);
    result = parallel_addition(inputMatrixA,inputMatrixB);
    writeMatrixToFile(outputFileName5, result);

    result = serial_subtraction(inputMatrixA,inputMatrixB);
    writeMatrixToFile(outputFileName6, result);
    result = parallel_subtraction(inputMatrixA,inputMatrixB);
    writeMatrixToFile(outputFileName7, result);

    cout << "------------------------------Matrix Multiplication---------------------------------------\n" << endl;
    cout << "          Serially using 1" << " thread. Time taken: " << exec_time[0] << "s"<< endl;
    cout << "          Parallel (not optimized) using " << num_threads << " threads. Time taken: " << exec_time[1] << "s" << endl;
    cout << "          Parallel (optimized) using " << num_threads << " threads. Time taken: " << exec_time[2] << "s" << endl;
    cout << "	  Parallel Speedup using " << num_threads << " threads is " << (exec_time[0]/exec_time[1]) << endl;  
    cout << "\n------------------------------MatrixA Transpose-------------------------------------------\n" << endl;
    cout << "          Serially using 1" << " thread. Time taken: " << exec_time[3] << "s" << endl;
    cout << "          Parallel using " << num_threads << " threads. Time taken: " << exec_time[10] << "s" << endl;
    cout << "	  Parallel Speedup using " << num_threads << " threads is " << (exec_time[3]/exec_time[10]) << endl;  

    if ((inputMatrixA.size() == inputMatrixB.size()) && (inputMatrixA[0].size() == inputMatrixB[0].size()))
    {
       result = serial_addition(inputMatrixA,inputMatrixB);
       writeMatrixToFile(outputFileName4, result);
       result = parallel_addition(inputMatrixA,inputMatrixB);
       writeMatrixToFile(outputFileName5, result);

       result = serial_subtraction(inputMatrixA,inputMatrixB);
       writeMatrixToFile(outputFileName6, result);
       result = parallel_subtraction(inputMatrixA,inputMatrixB);
       writeMatrixToFile(outputFileName7, result);

       cout << "\n---------------------------------Matrix Addition------------------------------------------\n" << endl;
       cout << "          Serially using 1" << " thread. Time taken: " << exec_time[4] << "s" << endl;
       cout << "          Parallel using " << num_threads << " threads. Time taken: " << exec_time[5] << "s" <<  endl;
       cout << "	  Parallel Speedup using " << num_threads << "threads is " << (exec_time[4]/exec_time[5]) << endl;  
       cout << "\n---------------------------------Matrix Subtraction---------------------------------------\n" << endl;
       cout << "          Serially using 1" << " thread. Time taken: " << exec_time[6] << "s" << endl;
       cout << "          Parallel using " << num_threads << " threads. Time taken: " << exec_time[7] << "s" << endl;
       cout << "	  Parallel Speedup using " << num_threads << " threads is " << (exec_time[6]/exec_time[7]) << endl;  
    }
    else cout << "Error: Matrix addition and subtraction need matrices with same dimensions" << endl;
    if (inputMatrixA.size() == inputMatrixA[0].size())
    {
       result = serial_inverse(inputMatrixA);
       writeMatrixToFile(outputFileName8, result);
       inputMatrixA = readMatrixFromFile(inputFileName1);
       result = parallel_inverse(inputMatrixA);
       writeMatrixToFile(outputFileName9, result);

       cout << "\n--------------------------------MatrixA Inversion-----------------------------------------\n" << endl;
       cout << "          Serially using 1" << " thread. Time taken: " << exec_time[8] << "s" << endl;
       cout << "          Parallel using " << num_threads << " threads. Time taken: " << exec_time[9] << "s" << endl;
       cout << "	  Parallel Speedup using " << num_threads << " threads is " << (exec_time[8]/exec_time[9]) << endl;  
       cout << "\n------------------------------------------------------------------------------------------" << endl;
    }
    else cout << "Error: Matrix Inversion requires a square matrix" << endl;

    return 0;
}

// Function to read matrix from a text file
vector<vector<double>> readMatrixFromFile(const char* filename) {
    ifstream file(filename);
    vector<vector<double>> matrix;
    if (file.is_open()) {
        int rows, cols;
        file >> rows >> cols;

        matrix.resize(rows, vector<double>(cols));

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int val;
                file >> val;
                matrix[i][j] = static_cast<double>(val);
            }
        }
        file.close();
    } else {
        cerr << "Error: Unable to open file" << endl;
        exit(1);
    }
    return matrix;
}

// Function to write matrix to a text file
void writeMatrixToFile(const char* filename, const vector<vector<double>>& result) {
    ofstream file(filename);
    if (file.is_open()) {
        file << result.size() << " " << result[0].size() << endl;
        for (const auto& row : result) {
            for (double val : row) {
                file << val << " ";
            }
            file << endl;
        }
        file.close();
    } else {
        cerr << "Error: Unable to open file" << endl;
        exit(1);
    }
}

/*
This method will get the transpose of matrix
*/
vector<vector<double>> serial_transpose(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<double>> trans_matrix(cols, vector<double>(rows, 0));

    double startTime = omp_get_wtime();

    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            trans_matrix[j][i] = matrix[i][j];
        }
    }

     //get the end time from the wall clock
     double endTime = omp_get_wtime();
     exec_time[3] = endTime - startTime;

    return trans_matrix;
}

/*
This method will get the transpose of matrix
*/
vector<vector<double>> parallel_transpose(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<double>> trans_matrix(cols, vector<double>(rows, 0));

    double startTime = omp_get_wtime();

    #pragma omp parallel for collapse(2) shared(matrix, trans_matrix) schedule(static)
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            trans_matrix[j][i] = matrix[i][j];
        }
    }

     //get the end time from the wall clock
     double endTime = omp_get_wtime();
     exec_time[10] = endTime - startTime;

    return trans_matrix;
}

vector<vector<double>> complexTransformation(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<double>> result(rows, vector<double>(cols, 0));

    // Apply a more complex transformation to each element of the matrix
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double temp = exp(sqrt(log(abs(matrix[i][j]) + 1.0))) * exp(sqrt(log(abs(matrix[i][j]) + 1.0)));
            result[i][j] = sin(temp) + cos(temp) * tan(temp);
        }
    }

    return result;
}

vector<vector<double>> serial_multiplication(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB) {
    int rows1 = matrixA.size();
    int cols1 = matrixA[0].size();
    int rows2 = matrixB.size();
    int cols2 = matrixB[0].size();

    // Apply a more complex transformation to matrices before multiplication
    vector<vector<double>> transformedA = complexTransformation(matrixA);
    vector<vector<double>> transformedB = complexTransformation(matrixB);

    vector<vector<double>> result(rows1, vector<double>(cols2, 0));

    // Start time from the wall clock
    double startTime = omp_get_wtime();

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < cols1; ++k) {
                result[i][j] += transformedA[i][k] * transformedB[k][j];
            }
        }
    }

    // Get the end time from the wall clock
    double endTime = omp_get_wtime();
    exec_time[0] = endTime - startTime;

    return result;
}

vector<vector<double>> complexTransformation_p(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<double>> result(rows, vector<double>(cols, 0));

    // Apply a more complex transformation to each element of the matrix
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double temp = exp(sqrt(log(abs(matrix[i][j]) + 1.0))) * exp(sqrt(log(abs(matrix[i][j]) + 1.0)));
            result[i][j] = sin(temp) + cos(temp) * tan(temp);
        }
    }

    return result;
}


vector<vector<double>> parallel_multiplication(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB){
        int rows1 = matrixA.size();
        int cols1 = matrixA[0].size();
        int rows2 = matrixB.size();
        int cols2 = matrixB[0].size();

        vector<vector<double>> result(rows1, vector<double>(cols2, 0));
        
         // Apply a more complex transformation to matrices before multiplication
     vector<vector<double>> transformedA = complexTransformation_p(matrixA);
     vector<vector<double>> transformedB = complexTransformation_p(matrixB);

	//start time from the wall clock
	double startTime = omp_get_wtime();
	//for loop spefication with shared, private variables and number of threads
	#pragma omp parallel for shared(transformedA,transformedB,result) schedule(static)
	for (int i = 0; i < rows1 ; i++ ){
		for (int j = 0; j < cols2 ; j++ ){
			result[i][j] = 0;
			for (int k = 0; k < cols1 ; k++ ){
				result[i][j] += transformedA[i][k]*transformedB[k][j];
			}
		}
	}

	//get the end time from wall clock
	double endTime = omp_get_wtime();

	//return the execution time
	exec_time[1] = endTime - startTime;
        return result;
}


vector<vector<double>> parallel_multiplication_op(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB) {
    int rows1 = matrixA.size();
    int cols1 = matrixA[0].size();
    int cols2 = matrixB[0].size();

    vector<vector<double>> result(rows1, vector<double>(cols2, 0));
    
      // Apply a more complex transformation to matrices before multiplication
     vector<vector<double>> transformedA = complexTransformation_p(matrixA);
     vector<vector<double>> transformedB = complexTransformation_p(matrixB);

    // Start time from the wall clock
    double startTime = omp_get_wtime();

    #pragma omp parallel for shared(transformedA, transformedB, result) schedule(static)
    for (int i = 0; i < rows1; ++i) {
        for (int k = 0; k < cols1; ++k) {
            int temp = transformedA[i][k]; // Cache matrixA element for efficiency
            for (int j = 0; j < cols2; ++j) {
                result[i][j] += temp * transformedB[k][j];
            }
        }
    }

    // Get the end time from the wall clock
    double endTime = omp_get_wtime();

    // Return the execution time 
    exec_time[2] = endTime - startTime;

    return result;
}

vector<vector<double>> serial_addition(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB) {
    int rows = matrixA.size();
    int cols = matrixA[0].size();

    vector<vector<double>> result(rows, vector<double>(cols, 0));

    // Start time from the wall clock
    double startTime = omp_get_wtime();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = matrixA[i][j] + matrixB[i][j];
        }
    }

    // Get the end time from the wall clock
    double endTime = omp_get_wtime();

    // Return the execution time (if needed)
    exec_time[4] = endTime - startTime;
    // Do something with exec_time if needed

    return result;
}

vector<vector<double>> parallel_addition(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB) {
    int rows = matrixA.size();
    int cols = matrixA[0].size();

    vector<vector<double>> result(rows, vector<double>(cols, 0));

    // Start time from the wall clock
    double startTime = omp_get_wtime();

    #pragma omp parallel for shared(matrixA, matrixB, result) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = matrixA[i][j] + matrixB[i][j];
        }
    }

    // Get the end time from the wall clock
    double endTime = omp_get_wtime();

    // Return the execution time (if needed)
    exec_time[5] = endTime - startTime;
    // Do something with exec_time if needed

    return result;
}

vector<vector<double>> serial_subtraction(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB) {
    int rows = matrixA.size();
    int cols = matrixA[0].size();

    vector<vector<double>> result(rows, vector<double>(cols, 0));

    // Start time from the wall clock
    double startTime = omp_get_wtime();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = matrixA[i][j] - matrixB[i][j];
        }
    }

    // Get the end time from the wall clock
    double endTime = omp_get_wtime();

    // Return the execution time (if needed)
    exec_time[6] = endTime - startTime;
    // Do something with exec_time if needed

    return result;
}

vector<vector<double>> parallel_subtraction(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB) {
    int rows = matrixA.size();
    int cols = matrixA[0].size();

    vector<vector<double>> result(rows, vector<double>(cols, 0));

    // Start time from the wall clock
    double startTime = omp_get_wtime();

    #pragma omp parallel for shared(matrixA, matrixB, result) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = matrixA[i][j] - matrixB[i][j];
        }
    }

    // Get the end time from the wall clock
    double endTime = omp_get_wtime();

    // Return the execution time (if needed)
    exec_time[7] = endTime - startTime;
    // Do something with exec_time if needed

    return result;
}

// Function to perform Gaussian elimination with partial pivoting
void gaussianElimination(vector<vector<double>>& matrix, int n) {
    for (int i = 0; i < n; ++i) {
        int maxIndex = i;
        double maxVal = abs(matrix[i][i]);

        // Partial pivoting
        for (int j = i + 1; j < n; ++j) {
            if (abs(matrix[j][i]) > maxVal) {
                maxVal = abs(matrix[j][i]);
                maxIndex = j;
            }
        }

        // Swap rows if needed for partial pivoting
        if (maxIndex != i) {
            swap(matrix[i], matrix[maxIndex]);
        }

        // Perform row operations to get identity matrix on the left
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                double factor = matrix[j][i] / matrix[i][i];
                for (int k = 0; k < n * 2; ++k) {
                    matrix[j][k] -= factor * matrix[i][k];
                }
            }
        }

        // Make the diagonal elements to 1
        double divisor = matrix[i][i];
        for (int j = 0; j < n * 2; ++j) {
            matrix[i][j] /= divisor;
        }
    }
}

// Function to calculate the inverse of a matrix
vector<vector<double>> serial_inverse(vector<vector<double>>& matrix) {
    int n = matrix.size();

    // Start time from the wall clock
    double startTime = omp_get_wtime();

    // Augment the matrix with an identity matrix
    for (int i = 0; i < n; ++i) {
        matrix[i].resize(2 * n);
        matrix[i][n + i] = 1.0;
    }

    // Perform Gaussian elimination with partial pivoting
    gaussianElimination(matrix, n);

    // Remove the original matrix and keep only the inverted part
    for (int i = 0; i < n; ++i) {
        matrix[i].erase(matrix[i].begin(), matrix[i].begin() + n);
    }

    // Get the end time from the wall clock
    double endTime = omp_get_wtime();

    // Return the execution time (if needed)
    exec_time[8] = endTime - startTime;

    return matrix;
}

void gaussianElimination1(vector<vector<double>>& matrix, int n) {
    #pragma omp parallel for shared(matrix, n) schedule(static)
    for (int i = 0; i < n; ++i) {
        int maxIndex = i;
        double maxVal = abs(matrix[i][i]);

        // Partial pivoting
        #pragma omp parallel for shared(matrix, i, maxIndex, maxVal, n)
        for (int j = i + 1; j < n; ++j) {
            double currentVal = abs(matrix[j][i]);
            #pragma omp critical
            {
                if (currentVal > maxVal) {
                    maxVal = currentVal;
                    maxIndex = j;
                }
            }
        }

        // Swap rows if needed for partial pivoting
        if (maxIndex != i) {
            #pragma omp critical
            swap(matrix[i], matrix[maxIndex]);
        }

        // Perform row operations to get identity matrix on the left
        #pragma omp parallel for shared(matrix, i, n)
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                double factor = matrix[j][i] / matrix[i][i];
                for (int k = 0; k < n * 2; ++k) {
                    matrix[j][k] -= factor * matrix[i][k];
                }
            }
        }

        // Make the diagonal elements 1
        double divisor = matrix[i][i];
        #pragma omp parallel for shared(matrix, i, divisor, n)
        for (int j = 0; j < n * 2; ++j) {
            matrix[i][j] /= divisor;
        }
    }
}

vector<vector<double>> parallel_inverse(vector<vector<double>>& matrix) {
    int n = matrix.size();

    // Start time from the wall clock
    double startTime = omp_get_wtime();

    #pragma omp parallel for shared(matrix, n)
    for (int i = 0; i < n; ++i) {
        matrix[i].resize(2 * n);
        matrix[i][n + i] = 1.0;
    }

    // Perform Gaussian elimination with partial pivoting
    gaussianElimination1(matrix, n);

    #pragma omp parallel for shared(matrix, n)
    for (int i = 0; i < n; ++i) {
        matrix[i].erase(matrix[i].begin(), matrix[i].begin() + n);
    }

    // Get the end time from the wall clock
    double endTime = omp_get_wtime();

    // Return the execution time (if needed)
    exec_time[9] = endTime - startTime;

    return matrix;
}

