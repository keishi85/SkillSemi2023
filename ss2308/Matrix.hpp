#include <iostream>
#include <array>

using Matrix3x3 = std::array<std::array<double, 3>, 3>;
using Vector3 = std::array<double, 3>;

Vector3 multiplyMatrixVector(const Matrix3x3& matrix, const Vector3& vector) {
    Vector3 result{};
    for (int i = 0; i < 3; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < 3; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

static Matrix3x3 multiplyMatrixMatrix(const Matrix3x3& matrix1, const Matrix3x3& matrix2) {
        Matrix3x3 result{};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result[i][j] = 0.0;
                for (int k = 0; k < 3; ++k) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return result;
}