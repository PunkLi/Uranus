/**
 * 模式识别 Knn
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-18
 */

#include <iostream>
#include <vector>
#include "uranus/Tensor.hpp"

#define Iris

#ifdef Iris
constexpr int Dim = 4;
std::string path = "../data/iris.data";
std::vector<int> data_class = { 50,50,50 };
#else
constexpr int Dim = 60;
std::string path = "../data/sonar.all-data";
std::vector<int> data_class = { 97,111 };
#endif

using sample = uranus::Tensor<Dim>::sampleType;
using sample_set = uranus::Tensor<Dim>::sample_set;
using tensor = uranus::Tensor<Dim>::TensorType;

int uniform_intx(int a, int b)
		{
			// linear_congruential_engine  线性同余法
			// mersenne_twister_engine     梅森旋转法
			// substract_with_carry_engine 滞后Fibonacci

			static std::default_random_engine e{ std::random_device{}() };
			static std::uniform_int_distribution<int> u;
			return u(e, std::uniform_int_distribution<int>::param_type(a, b));
		}

sampleType calu_mean(sample_set data_set)
{
    const int batch_size = data_set.size();
	sampleType vector_mean;
    setZero<sampleType, feature_rows>(vector_mean);

	for (int i = 0; i < batch_size; ++i) vector_mean += data_set[i];
	
	return vector_mean / batch_size;
}

int main()
{
    uranus::Data_Wrapper<Dim> wrapper(path, data_class);
	uranus::Tensor<Dim> data(wrapper, data_class);

	sample_set set1, set2, set3;

    sample mean1 = calu_mean(set1);
    sample mean2 = calu_mean(set2);
    sample mean3 = calu_mean(set3);

    double a = uranus::Norm<sample,4>(mean1, mean2);
    double b = uranus::Norm<sample,4>(mean1, mean3);
    double c = uranus::Norm<sample,4>(mean2, mean3);

    double J = a+b+c;

    int rand_num = uniform_intx(0,3);
    
    // {1,1,1,0};

    for(int i = 0; i< 100; ++i)
    {
        
    }



    
}