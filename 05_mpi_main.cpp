#include <iostream>
#include <vector>
#include <mpi.h>
#include <fstream>
#include <cstdlib> // Include for exit function
#include <math.h>
#include <cmath>

using namespace std;

double exec_time, exec_time1;

vector<vector<double>> copyMatrix(const vector<vector<double>>& source);
vector<vector<double>> readMatrixFromFile(const char* filename);
void writeMatrixToFile(const char* filename, const vector<vector<double>>& result);
vector<vector<double>> serial_multiplication(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB);
vector<vector<double>> parallel_multiplication( vector<vector<double>>& matrixA,  vector<vector<double>>& Input_matrixB);

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 2) {
        if (world_rank == 0) {
            cerr << "Usage: " << argv[0] << " <num_threads>" << endl;
        }
        MPI_Finalize();
        exit(1);
    }

    const char* inputFileName1 = "gen_matrixA.txt";
    const char* inputFileName2 = "gen_matrixB.txt";
    const char* outputFileName1 = "output_matrix_s_mul_mpi.txt";
    const char* outputFileName2 = "output_matrix_p_mul_mpi.txt";

    int num_threads = stoi(argv[1]);

    vector<vector<double>> inputMatrixA;
    vector<vector<double>> inputMatrixB;
    vector<vector<double>> result;

    if (world_rank == 0) {
        // Master process reads input matrices and distributes work
        inputMatrixA = readMatrixFromFile(inputFileName1);
        inputMatrixB = readMatrixFromFile(inputFileName2);

        if (inputMatrixA.empty() || inputMatrixB.empty()) {
            cerr << "Error: Empty matrix or file not found." << endl;
            MPI_Finalize();
            exit(1);
        }
    }

    // Broadcast input matrices to all processes
    MPI_Bcast(&inputMatrixA[0][0], inputMatrixA.size() * inputMatrixA[0].size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&inputMatrixB[0][0], inputMatrixB.size() * inputMatrixB[0].size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Serial matrix multiplication
    result = serial_multiplication(inputMatrixA, inputMatrixB);

    // Gather results to the master process
    MPI_Gather(&result[0][0], result.size() * result[0].size(), MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Master process writes the result to a file
    if (world_rank == 0) {
        writeMatrixToFile(outputFileName1, result);

        cout << "------------------------------Matrix Multiplication---------------------------------------\n" << endl;
        cout << "          Serially using 1 thread. Time taken: " << exec_time << "s" << endl;
    }

    // Parallel matrix multiplication
    result = parallel_multiplication(inputMatrixA, inputMatrixB);

    // Gather results to the master process
    MPI_Gather(&result[0][0], result.size() * result[0].size(), MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Master process writes the result to a file
    if (world_rank == 0) {
        writeMatrixToFile(outputFileName2, result);

        cout << "          Parallel (not optimized) using " << num_threads << " threads. Time taken: " << exec_time1 << "s" << endl;
        cout << "	  Parallel Speedup using " << num_threads << " threads is " << (exec_time/exec_time1) << endl; 
         cout << "\n-----------------------------------------------------------------------------------------\n" << endl; 
    }

    MPI_Finalize();
    return 0;
}

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
        MPI_Finalize();
        exit(1);
    }
    return matrix;
}

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
        MPI_Finalize();
        exit(1);
    }
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

    vector<vector<double>> result(rows1, vector<double>(cols2, 0));
    
    // Start time from the wall clock
    double startTime = MPI_Wtime();
    
    // Apply a more complex transformation to matrices before multiplication
     vector<vector<double>> transformedA = complexTransformation(matrixA);
     vector<vector<double>> transformedB = complexTransformation(matrixB);
     

    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < cols1; k++) {
                result[i][j] += transformedA[i][k] * transformedB[k][j];
            }
        }
    }

    // Get the end time from the wall clock
    double endTime = MPI_Wtime();
    exec_time = endTime - startTime;

    // Return the execution time
    return result;
}

vector<vector<double>> parallel_multiplication( vector<vector<double>>& mA,  vector<vector<double>>& mB) {
    int rows1 = mA.size();
    int cols1 = mA[0].size();
    int rows2 = mB.size();
    int cols2 = mB[0].size();

    vector<vector<double>> result(rows1, vector<double>(cols2, 0));
    vector<vector<double>> matrixA = complexTransformation(mA);
    vector<vector<double>> matrixB = complexTransformation(mB);


    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Broadcast matrixB to all processes
    MPI_Bcast(&matrixB[0][0], rows2 * cols2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Start time from the wall clock
    double startTime = MPI_Wtime();

    // Divide the rows of matrixA among processes
    int local_rows = rows1 / world_size;
    int start_row = world_rank * local_rows;
    int end_row = (world_rank == world_size - 1) ? rows1 : start_row + local_rows;

    // Local computation of result
    vector<vector<double>> local_result(local_rows, vector<double>(cols2, 0));
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < cols2; j++) {
            local_result[i - start_row][j] = 0;
            for (int k = 0; k < cols1; k++) {
                local_result[i - start_row][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }

    // Gather the partial results from all processes
    MPI_Gather(&local_result[0][0], local_rows * cols2, MPI_DOUBLE, &result[0][0], local_rows * cols2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Get the end time from the wall clock
    double endTime = MPI_Wtime();
    exec_time1 = endTime - startTime;

    return result;
}

