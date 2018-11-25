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

#include <Eigen/Dense>
#include <H5Cpp.h>

#include "eigen3-hdf5.hpp"
#include "gtest-helpers.hpp"

TEST(Attribute, Matrix) {
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> rmat1, rmat2;
    Eigen::Matrix<double, 2, 3, Eigen::ColMajor> cmat1, cmat2;
    rmat1 << 1, 2, 3, 4, 5, 6;
    cmat1 << 1, 2, 3, 4, 5, 6;
    {
        H5::H5File file("test_Attribute_Matrix.h5", H5F_ACC_TRUNC);
        EigenHDF5::save_attribute(file, "rowmat", rmat1);
        EigenHDF5::save_attribute(file, "colmat", cmat1);
    }
    {
        H5::H5File file("test_Attribute_Matrix.h5", H5F_ACC_RDONLY);
        EigenHDF5::load_attribute(file, "rowmat", rmat2);
        EigenHDF5::load_attribute(file, "colmat", cmat2);
    }
    ASSERT_PRED_FORMAT2(assert_same, rmat1, rmat2);
    ASSERT_PRED_FORMAT2(assert_same, cmat1, cmat2);
    ASSERT_PRED_FORMAT2(assert_same, rmat2, cmat2);
}

TEST(Attribute, Integer) {
    H5::H5File file("test_Attribute_Integer.h5", H5F_ACC_TRUNC);
    EigenHDF5::save_scalar_attribute(file, "integer", 23);
}

TEST(Attribute, Double) {
    H5::H5File file("test_Attribute_Double.h5", H5F_ACC_TRUNC);
    EigenHDF5::save_scalar_attribute(file, "double", 23.7);
}

TEST(Attribute, String) {
    H5::H5File file("test_Attribute_String.h5", H5F_ACC_TRUNC);
    EigenHDF5::save_scalar_attribute(file, "str1", std::string("hello"));
    EigenHDF5::save_scalar_attribute(file, "str2", "goodbye");
    const char *s = "again";
    EigenHDF5::save_scalar_attribute(file, "str3", s);
}
