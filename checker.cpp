#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

bool areMatricesEqual(const string& file1, const string& file2) {
    ifstream stream1(file1);
    ifstream stream2(file2);

    if (!stream1.is_open() || !stream2.is_open()) {
        cerr << "Error opening files." << endl;
        return false;
    }

    // Read matrix dimensions from the first line
    int rows1, cols1, rows2, cols2;
    stream1 >> rows1 >> cols1;
    stream2 >> rows2 >> cols2;

    // Check if matrix dimensions are the same
    if (rows1 != rows2 || cols1 != cols2) {
        return false;
    }

    // Read and compare matrices
    vector<vector<int>> matrix1(rows1, vector<int>(cols1));
    vector<vector<int>> matrix2(rows2, vector<int>(cols2));

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols1; ++j) {
            stream1 >> matrix1[i][j];
        }
    }

    for (int i = 0; i < rows2; ++i) {
        for (int j = 0; j < cols2; ++j) {
            stream2 >> matrix2[i][j];
        }
    }

    // Compare matrices
    return matrix1 == matrix2;
}

int main() {
    std::string file1, file2;

    cout << "Enter the path of the first text file: ";
    cin >> file1;

    cout << "Enter the path of the second text file: ";
    cin >> file2;

    if (areMatricesEqual(file1, file2)) {
        cout << "Both matrices in the files are the same." << endl;
    } else {
        cout << "Matrices in the files are different." << endl;
    }

    return 0;
}
