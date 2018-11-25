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

#include <iostream>

#include <Eigen/Dense>
#include <H5Cpp.h>

#include "eigen3-hdf5.hpp"

#include <gtest/gtest.h>

TEST(VectorRoundTrip, Double) {
    Eigen::Vector4d mat, mat2;
    mat << 1, 2, 3, 4;
#ifdef LOGGING
    std::cout << mat << std::endl;
#endif
    {
        H5::H5File file("test_VectorRoundTrip_Double.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "double_vector", mat);
    }
    {
        H5::H5File file("test_VectorRoundTrip_Double.h5", H5F_ACC_RDONLY);
        EigenHDF5::load(file, "double_vector", mat2);
    }
    ASSERT_EQ(mat, mat2);
}

TEST(VectorRoundTrip, Int) {
    Eigen::VectorXi mat(12), mat2;
    mat << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
#ifdef LOGGING
    std::cout << mat << std::endl;
#endif
    {
        H5::H5File file("test_VectorRoundTrip_Int.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "int_vector", mat);
    }
    {
        H5::H5File file("test_VectorRoundTrip_Int.h5", H5F_ACC_RDONLY);
        EigenHDF5::load(file, "int_vector", mat2);
    }
    ASSERT_EQ(mat, mat2);
}
