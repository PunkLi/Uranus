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
#include <gtest/gtest.h>

namespace Eigen {
    // C++ and/or gtest require that these two methods, which are used by calling
    // ASSERT_PRED_FORMAT2, be in the namespace of its argument. 

    // utility function to print an eigen object to an ostream; gtest will use this when
    // it outputs a matrix used in a failed assertion. Without this function, gtest seems
    // to dump some kind of byte representation of an eigen matrix, which is not very
    // helpful. 
    template <class Derived>
    void PrintTo(const Eigen::EigenBase<Derived>& mat, ::std::ostream* os)
    {
        (*os) << mat.derived() << "\n";
    }

    // utility function for gtest to use to check if two eigen objects are identical.
    // returns assertion success when they are identical; returns assertion failure along
    // with a nicely formatted message with the matrix contents when they are not
    // identical.
    // 
    // I put this function in this matrix test cpp file for a few reasons: 1) there is
    // not already a header file to put common test code for eigen3-hdf5, and 2) because
    // I needed it to help me debug test failures as I implemented the no copy read and
    // write functions. I really think that providing a header for common test code
    // should be addressed at some point, and then this function (and its companion
    // PrintTo) should be moved there.
    // 
    // Usage:
    // 
    // ASSERT_PRED_FORMAT2(assert_same, mat, mat2); 
    template <class DerivedExp, class DerivedAct>
    ::testing::AssertionResult assert_same(const char* exp_expr,
        const char* act_expr,
        const Eigen::EigenBase<DerivedExp>& exp,
        const Eigen::EigenBase<DerivedAct>& act)
    {
        if (exp.rows() == act.rows() &&
            exp.cols() == act.cols() &&
            exp.derived() == act.derived())
        {
            return ::testing::AssertionSuccess();
        }

        // if eigen did not define the == operator, you could use
        // exp.derived().cwiseEqual(act.derived()).all();

        ::testing::AssertionResult result = ::testing::AssertionFailure()
            << "Eigen objects are not the same: ("
            << exp_expr << ", " << act_expr << ")\n"
            << exp_expr << ":\n"
            << ::testing::PrintToString(exp)
            << "\n---and\n" << act_expr << ":\n"
            << ::testing::PrintToString(act)
            << "\n---are not equal!\n";

        return result;
    }
} // namespace Eigen
