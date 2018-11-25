// The MIT License (MIT)

// Copyright (c) 2013 James R. Garrison

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <complex>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <H5Cpp.h>

#include "eigen3-hdf5-sparse.hpp"

#include <gtest/gtest.h>

TEST(SparseMatrix, Double) {
    Eigen::SparseMatrix<double> mat(3, 3), mat2;
    mat.insert(0, 1) = 2.7;
    mat.insert(2, 0) = 82;
    {
        H5::H5File file("test_SparseMatrix_Double.h5", H5F_ACC_TRUNC);
        EigenHDF5::save_sparse(file, "mat", mat);
    }
    {
        H5::H5File file("test_SparseMatrix_Double.h5", H5F_ACC_RDONLY);
        EigenHDF5::load_sparse(file, "mat", mat2);
    }
#ifdef LOGGING
    std::cout << mat2 << std::endl;
#endif
    ASSERT_EQ(Eigen::MatrixXd(mat), Eigen::MatrixXd(mat2));
}

TEST(SparseMatrix, Complex) {
    Eigen::SparseMatrix<std::complex<double> > mat(4, 4), mat2;
    mat.insert(0, 1) = std::complex<double>(2, 4.5);
    mat.insert(1, 2) = std::complex<double>(82, 1);
    {
        H5::H5File file("test_SparseMatrix_Complex.h5", H5F_ACC_TRUNC);
        EigenHDF5::save_sparse(file, "mat", mat);
    }
    {
        H5::H5File file("test_SparseMatrix_Complex.h5", H5F_ACC_RDONLY);
        EigenHDF5::load_sparse(file, "mat", mat2);
    }
#ifdef LOGGING
    std::cout << mat2 << std::endl;
#endif
    ASSERT_EQ(Eigen::MatrixXcd(mat), Eigen::MatrixXcd(mat2));
}
