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

void Evaluation(const uranus::Vector<1> W0,
	const std::vector<uranus::Vector<1>>& set)
{
	size_t size = set.size();
	double P = 0, N = 0;
	for (int i = 0; i < size; ++i)
	{
		double sub = set[i](0) - W0(0);
		if (sub > 0)
			P++;
		else
			N++;
	}
	double rate = (P / size) > (N / size) ? (P / size) : (N / size);

	std::cout << "rate:" << rate * 100 << "% \n";
}
