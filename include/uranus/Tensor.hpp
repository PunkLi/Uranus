// The MIT License (MIT)

// Copyright (c) 2018 li chunpeng, Xidian university

// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>
#include <random>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>

#include "Matrix.hpp"

namespace uranus
{
	/**
	 * @class uranus::Data_Wrapper
	 * @brief read data form file
	 */
	template<int feature_rows> class Data_Wrapper
	{
		const int sample_num;
	public:
		using DataSet = std::vector<        // vector of 
			std::pair<std::vector<double>,  // feature cols [维度]
			std::string                     // label
			> >;
		DataSet buffer;

		Data_Wrapper(const std::string file_name,
			const std::vector<int> vec,
			bool ishow = false)
			:sample_num(accumulate(vec.begin(), vec.end(), 0))
		{
			readData(file_name, buffer, feature_rows, sample_num, ishow);
		}

		/**
		 * @brief read data set from file ,such as *.csv & *.txt & *.data
		 * @param file_name
		 * @param feature_vec of feature
		 * @param feature_vec of sample set
		 * @param visual option
		 */
		void readData(const std::string& file_name,
			DataSet& buffer,
			const int feature_num,
			const int sample_num,
			bool visual = false)
		{
			buffer.resize(sample_num);
			std::ifstream fin(file_name);
			std::string line;

			for (int index = 0; index < sample_num; ++index)
			{
				std::getline(fin, line);
				std::istringstream sin(line);
				std::vector<std::string> fields;
				std::string field;
				while (std::getline(sin, field, ','))
					fields.push_back(field);

				int idx = feature_num;
				std::vector<double> feature_vec(fields.size());
				for (int idx = 0; idx < feature_num; ++idx)
					feature_vec[idx] = std::stod(Trim(fields[idx]));
				std::string label = Trim(fields[idx]);

				buffer[index] = std::make_pair(feature_vec, label);

				if (visual)
				{
					std::cout << "In buffer: ";
					for (int i = 0; i < feature_num; ++i)
						std::cout << feature_vec[i] << "\t";
					std::cout << std::endl;
				}
			}
		}
	private:
		/**
		* @brief clean string & trash useless char elements
		* @param string
		* @return string
		*/
		std::string Trim(std::string& str)
		{
			str.erase(0, str.find_first_not_of(" \t\r\n"));
			str.erase(str.find_last_not_of(" \t\r\n") + 1);
			return str;
		}
	};

	/**
	* @class uranus::Tensor
	* @brief vector of class , vector of sample, vector of sample Dim,
	* such as [[[],[],[]], [[],[],[]], [[],[],[]] ]
	*/
	template<int feature_rows> class Tensor
	{
		std::vector<int> vec_size;
	public:
		using DataSet = std::vector<        // vector of 
			std::pair<std::vector<double>,  // feature cols [维度]
			std::string                     // label
			> >;

		// using Type
		using sampleType = uranus::Vector<feature_rows>;
		using sample_set = std::vector<sampleType>;
		using TensorType = std::vector<sample_set>;

		TensorType tensor;
		// Tensor(){}

		Tensor(const Data_Wrapper<feature_rows>& wrapper,
			const std::vector<int> class_size,
			bool ishow = false)
		{
			getfromfile(wrapper, class_size, ishow);
		}

		/**
		 * @brief get buffer data form data_wrapper
		 * @param reference of data_wrapper
		 * @param visual option
		 */
		void getfromfile(const Data_Wrapper<feature_rows>& wrapper,
			const std::vector<int> class_size,
			bool visual = false)
		{
			this->vec_size.assign(class_size.begin(), class_size.end());
			int size = class_size.size();
			tensor.resize(feature_rows);

			vec2vec(wrapper.buffer, feature_rows,  // buffer 
				tensor, 0,                     // tensor for 0
				0, class_size[0]);             // [low, hi

			for (int i = 1; i < size; ++i)
				vec2vec(wrapper.buffer, feature_rows,               // buffer
					tensor, i,                                  // tensor for i 
					_sum(class_size, i - 1), _sum(class_size, i));  // [low, hi]

			if (visual)
			{
				std::cout << "\n\n";
				for (int i = 1; i < size; ++i) output(class_size, i);
			}
		}
		/**
		 * @brief leave-one-out cross validation
		 * @param index of vector of class, such as vec[0] as {50} in {50,50,50}
		 * @param visual option
		 * @return Tensor::TensorType x_set {{train},{test}}
		 */
		TensorType leave_one_out_validation(const int index, bool visual = false)
		{
			TensorType x_set(2);
			int total = vec_size[index];
			int rand = uniform_intx(0, total - 1);
			for (int i = 0; i < total; ++i)
			{
				if (i != rand)
					x_set[0].push_back(this->tensor[index][i]);  // vec_size[index].size() - 1 of train
				else
					x_set[1].push_back(this->tensor[index][i]);  // one test sample
			}
			if (visual) {
				std::cout << "\n------Leave one out Validation------\n\n" 
						  << "Rand idx:" << rand
						  << "\t trian_set size: " << x_set[0].size() << "\t"
						  << "test_set size: " << x_set[1].size()
						  << "\n\n-----------------------------------\n";
			}
			return x_set;
		}
		/**
		 * @brief k-fold cross validation
		 * @param index of vector of class, such as vec[0] as {50} in {50,50,50}
		 * @param visual option
		 * @return Tensor::TensorType x_set {{train},{test}}
		 */
		template<int const_K>
		TensorType k_fold_crossValidation(const int index, bool visual = false)
		{
			if (visual)
				std::cout << "\n------" << const_K << "-fold cross Validation------\n";
				
			const int total = vec_size[index];
			const int batch_size = vec_size[index] / const_K;
			const int last_batch_idx = (const_K - 1)*batch_size;

			bool *hit_table = new bool[total];      // hit total
			int *last_batch = new int[batch_size];  // last batch

			for (int i = 0; i < total; ++i) *(hit_table + i) = 0;  // all false

			TensorType x_set(2);

			bool isHit = false;
			int rand_idxTable, idx_k = 0, has_hit = 0, test_size = 0;
			
			for (int i = 0; i < total; ++i)
			{
				do {
					if (has_hit >= last_batch_idx) // k-1 has done
					{
						if (visual)
						{
							std::cout << "\nhit_table:[";
							for (int i = 0; i < total; ++i)
								std::cout << *(hit_table + i);
							std::cout << "]  hit_idx: " << (const_K - 1)*batch_size << "\n\n";
						}
						for (int i = 0; i < total; ++i)
						{
							int _isHit = *(hit_table + i);
							if (!_isHit)
								*(last_batch + test_size++) = i;
						}
						goto double_break; // break do{...}while, then break for{...}
					}
					rand_idxTable = uniform_intx(0, total - 1);
					isHit = *(hit_table + rand_idxTable);       
					if (!isHit) 
						if (++has_hit >= total) break;
				} while (isHit);

				*(hit_table + rand_idxTable) = true;
				if (visual) 
					std::cout //<< i << ": "
							  << rand_idxTable << "\t";

				if ((i + 1) % batch_size == 0) // 取下一个k折
				{
					idx_k++;
					if (visual) std::cout <<"k-fold  of "<< idx_k << std::endl;
				}
				x_set[0].push_back(this->tensor[index][rand_idxTable]);
			} 
		double_break: // test set
			for (int i = 0; i < test_size; ++i)
			{
				int idx_rand = *(last_batch + i);
				if (visual)
					std::cout << idx_rand << "\t";
				x_set[1].push_back(this->tensor[index][idx_rand]);  // last batch 
			}
			if (visual) {
				std::cout << "\n-----------------------------------\n"
						  << "trian_set size: " << x_set[0].size() << "\n"
						  << "test_set size: " << x_set[1].size()
						  << "\n-----------------------------------\n";
			}
			return x_set; // 分成k折的vector<sampleType> 类型 Tensor
		}
	private:
		/**
		 * @brief generate random number int [a,b]
		 * @param  low bounder a
		 * @param high bounder b
		 * @return randon number in [a,b]
		 */
		int uniform_intx(int a, int b)
		{
			// linear_congruential_engine  线性同余法
			// mersenne_twister_engine     梅森旋转法
			// substract_with_carry_engine 滞后Fibonacci

			static std::default_random_engine e{ std::random_device{}() };
			static std::uniform_int_distribution<int> u;
			return u(e, std::uniform_int_distribution<int>::param_type(a, b));
		}

		/**
		 * @brief trans form std::vector to uranus::vector
		 * @param DataSet, which is buffer read from files
		 * @param feature_vec of feature
		 * @param vector of vector of uranus::vector , means tensor
		 * @param index of tensor
		 * @param index of begin iter, in buffer
		 * @param index of end iter, in buffer
		 */
		void vec2vec(
			const DataSet & buffer,
			const int featrue,
			std::vector<std::vector<sampleType>> & tensor,
			const int index,
			const int begin,
			const int end)
		{
			// To-do : 
			// std::vector -> uranus::vector , step by step 
			// std::vector<  std::pair<   std::vector<double>,  std::string  >>;
			// std::vector<             uranus::Vector<feature>              >> x_set;
			sampleType temp_set;
			for (int i = begin; i < end; ++i)
			{
				for (int j = 0; j < featrue; ++j)
					temp_set(j) = buffer[i].first[j];

				tensor[index].push_back(temp_set);
			}
		}

		int _sum(const std::vector<int>& class_size, int idx)
		{
			int sum = 0;
			for (int i = 0; i <= idx; ++i) sum += class_size[i]; // 这里是 <=
			return sum;
		}

		void output(const std::vector<int>& class_size, int idx)
		{
			for (int i = 0; i < class_size[idx]; ++i)
				std::cout << tensor[idx][i] << std::endl << std::endl;
		}
	};

}