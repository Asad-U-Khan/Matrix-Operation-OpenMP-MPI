#include <iostream>
#include <fstream>
#include <vector>
#include <random>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <M> <N>" << endl;
        return 1;
    }

    int M1 = stoi(argv[1]);
    int N1 = stoi(argv[2]);
    int M2 = stoi(argv[3]);
    int N2 = stoi(argv[4]);

    if (N1 != M2) {
        cerr << "Error: MatrixA col should be eq to MatrixB rows" << endl;
        return 1;
    }

    ofstream file("gen_matrixA.txt");
    if (file.is_open()) {
        file << M1 << " " << N1 << endl; // Write dimensions to the file

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dist(1, N1);

        for (int i = 0; i < M1; ++i) {
            for (int j = 0; j < N1; ++j) {
                file << dist(gen) << " ";
            }
            file << endl;
        }

        file.close();
        cout << "Matrix A of size " << M1 << "x" << N1 << " generated and stored in gen_matrixA.txt" << endl;
    } else {
        cerr << "Error: Unable to create file" << endl;
    }

    ofstream file1("gen_matrixB.txt");
    if (file1.is_open()) {
        file1 << M2 << " " << N2 << endl; // Write dimensions to the file

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dist(1, N2);

        for (int i = 0; i < M2; ++i) {
            for (int j = 0; j < N2; ++j) {
                file1 << dist(gen) << " ";
            }
            file1 << endl;
        }

        file1.close();
        cout << "Matrix B of size " << M2 << "x" << N2 << " generated and stored in gen_matrixB.txt" << endl;
    } else {
        cerr << "Error: Unable to create file" << endl;
    }

    return 0;
}
