/**
 * 模式识别 Fisher
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-17
 */

// 前言：二分类 && 齐次

//    const int N = 4;

//    Vector<N> x; // [x1,x2,x3,1]
//    Vector<N> y; // [x1,x2,x3,1]

//    Matrix<1,N> W; // w1,w2,w3,W_n-1

//    y = W*x; // 线性判别函数

     // 样本向量
//    template<int N> using sample_x = Vector<N>;
    // 权值向量
//    template<int N> using vec_w = Vector<N>;

    // 决策面 g(x)=0  // g(x) = g_1(x) - g_2(x)

    // x到决策面 H 的距离

//    auto r = g(x) / abs(W);  // 经过一些推导

    // 多分类问题

    // 1.每一类别可用单个判别边界与其它类别相分开，各自确定边界
    // 联立线性方程组

//    Matrix<N,N> A;

//    y = A*x;  // solve  -> 


    // 2.每个模式类和其它模式类间可分别用判别平面分开  
    // 这样 有 M(M - 1)/2个判别平面

    // 判别函数： g_{ij}(X) = w_{ij}^Tx
    // 判别边界： g_{ij}(x) = 0
    
    // g_{ij}(x)>0, x\in \oemga_1   < 0, x \in \omega_2

    // 结论：判别区间增大，不确定区间减少，比第一种情况小得多

    // 3. 每类都有一个判别函数, 存在M个判别函数

    // 判别函数 g(x)=W_K^TX ,k=1,2,...,M
    // 判别规则 g_i(x)=W_i^TX , Max: x\in\omeag_1 , samll other
    // 判别边界：g_i(x)=g_j(x) or g_i(x)-g_j(x)=0  
    // 优点：考虑了相邻的判别函数，可以保证交于一点
    // 不确定区间没有了，所以这种是最好情况


    // 广义线性判别函数
    // g(x) = w1f1+w2f2+w3f3+......+wk 
    // 这样一个非线性判别函数通过映射，变换成线性判别函数。
    // 原始的特征空间是非线性，
    // 但通过某种映射，在新的空间 能保证是线性函数
    // 原始空间的判别函数为广义线性判别函数。


// 为了降维，降低计算复杂度
// 易于分类

// 使两类样本在该轴上投影之间的距离尽可能远，
// 而每一类 样本的投影尽可能紧凑。如何度量

// 评价标准 — 类内离散度矩阵
//           类间离散度矩阵

#include <iostream>
#include <vector>
#include "uranus/Matrix.hpp"
#include "uranus/Function.hpp"

int main(int argc, char *argv[])
{
	using namespace std;
	const int batch_size = 10;

	int size = 0;
	int size2 = 0;
	constexpr int Dim = 3;

	std::vector<uranus::Vector<Dim>> x_set1(batch_size);  // 10组，3维数，class1
	std::vector<uranus::Vector<Dim>> x_set2(batch_size);  // 10组，3维数，class2

	x_set1[0] << -0.4, 0.58, 0.089;
	x_set1[1] << -0.31, 0.27, -0.04;
	x_set1[2] << -0.38, 0.055, -0.035;
	x_set1[3] << -0.15, 0.53, 0.011;
	x_set1[4] << -0.35, 0.47, 0.034;
	x_set1[5] << 0.17, 0.69, 0.1;
	x_set1[6] << -0.011, 0.55, -0.18;
	x_set1[7] << -0.27, 0.61, 0.12;
	x_set1[8] << -0.065, 0.49, 0.0012;
	x_set1[9] << -0.12, 0.054, -0.063;

	x_set2[0] << 0.83, 1.6, -0.014;
	x_set2[1] << 1.1, 1.6, 0.48;
	x_set2[2] << -0.44, -0.41, 0.32;
	x_set2[3] << 0.047, -0.45, 1.4;
	x_set2[4] << 0.28, 0.35, 3.1;
	x_set2[5] << -0.39, -0.48, 0.11;
	x_set2[6] << 0.34, -0.079, 0.14;
	x_set2[7] << -0.3, -0.22, 2.2;
	x_set2[8] << 1.1, 1.2, -0.46;
	x_set2[9] << 0.18, -0.11, -0.49;

	//cout << x_set1[0];

	uranus::Vector<Dim> mean_1;
	mean_1 << 0, 0, 0;
	uranus::Vector<Dim> mean_2;
	mean_2 << 0, 0, 0;
	uranus::SquareMatrix<Dim> Si_1;
	Si_1 << 0, 0, 0, 0, 0, 0, 0, 0, 0;
	uranus::SquareMatrix<Dim> Si_2;
	Si_2 << 0, 0, 0, 0, 0, 0, 0, 0, 0;
	
	// step1 均值向量
	for (int i = 0; i < batch_size; ++i)
	{
		mean_1 += x_set1[i]*0.1;
		mean_2 += x_set2[i]*0.1;
	}

	// step2 类内离散度矩阵
	for (int i = 0; i < batch_size; ++i)
	{
		Si_1 += (x_set1[i] - mean_1)*(x_set1[i] - mean_1).transpose();
		Si_2 += (x_set2[i] - mean_2)*(x_set2[i] - mean_2).transpose();
	}
	cout << "Si_1=\n"<< Si_1 << endl << endl;
	cout << "Si_2=\n"<< Si_2 << endl << endl;

	// step3
	// 总样本类内离散度矩阵Sw  对称半正定矩阵，而且当n>d时通常是非奇异的
	uranus::SquareMatrix<Dim> Sw = Si_1 + Si_2;
	cout << "Sw=\n" << Sw << endl << endl;

	// step4
	// 样本类间离散度矩阵SB
	uranus::SquareMatrix<Dim> Sb = (mean_1 - mean_2) * (mean_1 - mean_2).transpose();

	// step5
	// Fisher准则函数 -- 最佳投影方向
	// uranus::SquareMatrix<Dim> Jw = Sb*Sw.inverse();
	// w* = \argmax J(w)

	uranus::Vector<Dim> argW = Sw.inverse()*(mean_1 - mean_2);
	cout << "argW=\n" << argW << endl << endl;


	// step6求阈值 W0 
	uranus::Vector<1> W0 = argW.transpose()*mean_1 / 2 + argW.transpose()*mean_2 / 2;
	cout << "Wo=\n" << W0 << endl << endl;

	// step7线性变换
	std::vector<uranus::Vector<1>> D1(batch_size);
	std::vector<uranus::Vector<1>> D2(batch_size);
	for (int i = 0; i < batch_size; ++i)
	{
		D1[i] = argW.transpose()*x_set1[i];
		cout << D1[i] << "       ";
		D2[i] = argW.transpose()*x_set2[i];
		cout << D2[i] << endl;
	}
	return EXIT_SUCCESS;
}