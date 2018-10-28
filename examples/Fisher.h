
#ifndef _URANUS_FISHER_H_
#define _URANUS_FISHER_H_

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

template<int dim>
bool Multi_Discriminant(uranus::Vector<dim> plane_argW_1,
			   			uranus::Vector<dim> plane_argW_2,
						uranus::Vector<1> plane_W0_1,
						uranus::Vector<1> plane_W0_2,
						uranus::Vector<dim> var_x)
{
	uranus::Vector<dim> W = plane_argW_1 - plane_argW_2;
	double Wo = plane_W0_1(0) - plane_W0_2(0);
	uranus::Vector<1> Wx = W.transpose() * var_x;
	double result = Wx(0) + Wo;
	if (result > 0)
		return true;
	else
		return false;
};

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

#endif // _URANUS_FISHER_H_