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
#include "uranus/Tensor.hpp"
#include "Fisher.h"

constexpr int feature_rows = 3;
std::vector<int> data_class = { 10,10 };
std::string path = "../data/fisher-example";
// k折交叉验证
constexpr int K = 10;

std::vector<uranus::Vector<feature_rows>> mean;
std::vector<uranus::SquareMatrix<feature_rows>> Si;
uranus::SquareMatrix<feature_rows> Sw;


constexpr int Dim = feature_rows;

uranus::SquareMatrix<Dim> Si_1;
uranus::SquareMatrix<Dim> Si_2;

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
	
	std::cout << "rate:" << rate*100 << "% \n";
}

int main(int argc, char *argv[])
{
	using namespace std;
	const int batch_size = 10;

	using namespace std;
	using sample_set = uranus::Tensor<feature_rows>::sample_set;
	using tensor = uranus::Tensor<feature_rows>::TensorType;
	
	uranus::Data_Wrapper<feature_rows> wrapper(path, data_class, true);
	uranus::Tensor<feature_rows> data(wrapper, data_class);

	Si_1 << 0, 0, 0, 0, 0, 0, 0, 0, 0;
	Si_2 << 0, 0, 0, 0, 0, 0, 0, 0, 0;

	tensor tensor_x1 = data.k_fold_crossValidation<3>(0, true);
	tensor tensor_x2 = data.k_fold_crossValidation<4>(1, true);

	sample_set train_x1 = tensor_x1[0];
	sample_set train_x2 = tensor_x2[0];

	sample_set test_x1 = tensor_x1[1];
	sample_set test_x2 = tensor_x2[1];

	// step1 均值向量
	auto mean_0 = data.get_mean(train_x1, true);
	auto mean_1 = data.get_mean(train_x2, true);
	
	// step2 类内离散度矩阵
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < train_x1.size(); ++i)
			Si_1 += (train_x1[i] - mean_0)*(train_x1[i] - mean_0).transpose();
		cout << "Si_1=\n" << Si_1 << endl << endl;
#pragma omp for
		for (int i = 0; i < train_x2.size(); ++i)
			Si_2 += (train_x2[i] - mean_1)*(train_x2[i] - mean_1).transpose();
		cout << "Si_2=\n" << Si_2 << endl << endl;
	}

	// step3
	// 总样本类内离散度矩阵Sw  对称半正定矩阵，而且当n>d时通常是非奇异的
	uranus::SquareMatrix<Dim> Sw = Si_1 + Si_2;
	cout << "Sw=\n" << Sw << endl << endl;

	// step4
	// 样本类间离散度矩阵SB
	init_Sb_(Dim, mean_0, mean_1);
	cout << "Sb=\n" << Sb_(mean_0, mean_1) << endl << endl;
	// step5
	// Fisher准则函数 -- 最佳投影方向
	// uranus::SquareMatrix<Dim> Jw = Sb*Sw.inverse();
	// w* = \argmax J(w)
	init_argW_(Dim, mean_0, mean_1);
	cout << "argW=\n" << argW_(mean_0, mean_1) << endl << endl;

	// step6求阈值 W0 
	init_W0_(Dim, mean_0, mean_1);
	cout << "Wo=\n" << W0_(mean_0, mean_1) << endl << endl;

	// step7线性变换
	std::vector<uranus::Vector<1>> D1(test_x1.size());
	std::vector<uranus::Vector<1>> D2(test_x2.size());

	for (int i = 0; i < test_x1.size(); ++i)
	{
		D1[i] = argW_(mean_0, mean_1).transpose()*test_x1[i];
		cout << D1[i] << endl;
	}
	cout << "\n";
	for (int i = 0; i < test_x2.size(); ++i)
	{
		D2[i] = argW_(mean_0, mean_1).transpose()*test_x2[i];
		cout << D2[i] << endl;
	}
	
	Evaluation(W0_(mean_0, mean_1), D1);
	Evaluation(W0_(mean_0, mean_1), D2);
	
	return EXIT_SUCCESS;
}