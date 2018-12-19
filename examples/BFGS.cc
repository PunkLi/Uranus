#include <iostream>
#include <Eigen/Dense>
#include "uranus/Matrix.hpp"
#include "uranus/un-constrained.hpp"

int main()
{
	problem<2> f;	  // f = x_1^2 + 4x_2^2

	f.matirx_Jacobian_ << 2, 0, 0, 8; 

    uranus::Vector<2> x0; 
	x0 << 1, 1;                   // 初始点

	uranus::Vector<2> y = BFGS<2>(f,x0,0.001,true); // 用BFGS求解

    std::cout <<"zuiyoujie:\n" << y <<"\n";

    // dasds
    problem<2> f2;
    f2.matirx_Jacobian_ << 2, 0, 0, 2;
    uranus::Vector<2> x2;
    x2 << 5,5;

    double M_k = 0.1;                          // init M_k > 0	
    uranus::Vector<2> x3;
    double a;
	do{
		M_k = 10 * M_k;                         // update M_k
		f2.matirx_Jacobian_(1,1) = 2 + M_k;     // update J mat
        cout << "M_k=============================\n"<<M_k<<"\n";
		x3 = BFGS(f2, x2, 0.001, true);       // min f

        a = M_k * pow(x3(1)-1, 2);
        std::cout <<"x3: \n" << x3 <<"\n";
	}
	while(a > 0.01);

    std::cout <<"x3: \n" << x3 <<"\n";

    return 0;
}