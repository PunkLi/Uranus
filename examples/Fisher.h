#pragma once
#include <vector>
#include "uranus/Matrix.hpp"

#define init_Sb_(feature_rows, str1, str2)                \
	uranus::SquareMatrix<feature_rows> Sb_##str1##str2    \
		= ((str1) - (str2))                               \
		* ((str1) - (str2)).transpose();

#define Sb_(str1, str2)    Sb_##str1##str2  

#define init_argW_(feature_rows, str1, str2)              \
	uranus::Vector<feature_rows> argW_##str1##str2        \
		= Sw.inverse()*((str1) - (str2));             

#define argW_(str1, str2)  argW_##str1##str2 

#define init_W0_(feature_rows, str1, str2)                \
	uranus::Vector<1> W0_##str1##str2                     \
		= argW_(str1,str2).transpose()*(str1) / 2         \
		+ argW_(str1,str2).transpose()*(str2) / 2;

#define W0_(str1, str2)    W0_##str1##str2

/*
template<int Dim> class Fisher
{
	// Projection method
	void projection (uranus::Vector<feature_rows> set)
	{

	}
};
*/
