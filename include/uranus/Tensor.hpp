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
#include <list>

#include "Matrix.hpp"

namespace uranus
{
	// init
	constexpr int sample_num = 150;      // 样本数量
	constexpr int feature_rows = 4;      // feature 维数
	constexpr int class_cols = 3;        // class   类数
	constexpr int class_1 = 50;
	constexpr int class_2 = 50;
	constexpr int class_3 = 50;          // 这个位置需要进一步泛化

    /**
	 * @class uranus::Data_Wrapper
	 * @brief read data form file
	 */ 
	class Data_Wrapper
	{
	public:
		using DataSet = std::vector<                    // vector of 
						std::pair<std::vector<double>,  // feature cols [维度]
						std::string                     // label
						> >;
		DataSet buffer;

		Data_Wrapper(std::string file_name)
		{		
			readData(file_name, buffer, feature_rows, sample_num);
		}

		/**
		 * @brief read data set from file ,such as *.csv & *.txt & *.data
		 * @param file_name
		 * @param feature_vec of feature
		 * @param feature_vec of sample set
		 */
		void readData(std::string& file_name,
				  	  DataSet& buffer,
					  int feature_num,
					  int sample_num )
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

				std::cout << "处理之后的字符串："
					<< feature_vec[0] << "\t"
					<< feature_vec[1] << "\t"
					<< feature_vec[2] << "\t"
					<< feature_vec[3] << "\t"
					<< std::endl;
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
	class Tensor
	{
	public:
		using DataSet = std::vector<                    // vector of 
						std::pair<std::vector<double>,  // feature cols [维度]
						std::string                     // label
						> >;
			// init
			using sample = uranus::Vector<feature_rows>;

			// 这是可以直接用在fisher里头的
			std::vector<std::vector<sample>> tensor;
		Tensor(){}
		Tensor(Data_Wrapper& wrapper)
		{
			getfromfile(wrapper);
		}
        /**
         * @brief get buffer data form data_wrapper
         * @param reference of data_wrapper
         */
		void getfromfile(Data_Wrapper& wrapper)
		{
			tensor.resize(feature_rows);
			vec2vec(wrapper.buffer, feature_rows, tensor, 0, 0,   class_1);
			vec2vec(wrapper.buffer, feature_rows, tensor, 1, class_1,  class_1+class_2);
			vec2vec(wrapper.buffer, feature_rows, tensor, 2, class_1 + class_2, class_1 + class_2 + class_3);

			std::cout << "\n\n";
			for (int i = 0; i < class_1; ++i)
				std::cout << "[" << 0 << i << "]" << std::endl <<  tensor[0][i] << std::endl << std::endl;
			for (int i = 0; i < class_3; ++i)
				std::cout << "[" << 1 << i << "]" << std::endl << tensor[1][i] << std::endl << std::endl;
			for (int i = 0; i < class_3; ++i)
				std::cout << "["<<2 << i <<"]"<< std::endl << tensor[2][i] << std::endl << std::endl;
		}
			
	private:
		/**
		 * @brief trans form std::vector to uranus::vector
		 * @param DataSet, which is buffer read from files
		 * @param feature_vec of feature
		 * @param vector of vector of uranus::vector , means tensor
		 * @param index of tensor
		 * @param index of begin iter, in buffer
		 * @param index of end iter, in buffer
		 */
		void vec2vec(DataSet & buffer,
					int featrue,
					std::vector<std::vector<sample>> & tensor,
					int index,
					int begin,
					int end)
		{
			// To-do : 
			// std::vector -> uranus::vector , step by step 
			// std::vector<  std::pair<   std::vector<double>,  std::string  >>;
			// std::vector<             uranus::Vector<feature>              >> x_set;
			sample temp_set;
			for (int i = begin; i < end; ++i)
			{
				for (int j = 0; j < featrue; ++j)
					temp_set(j) = buffer[i].first[j];

				tensor[index].push_back(temp_set);
			}
		}
	};

}

	/**
	 * @brief generate random number int [a,b]
	 * @param  low bounder a
	 * @param high bounder b
	 * @return randon number in [a,b]
	 */
	int uniform_intx(int a, int b) {
		static std::default_random_engine e{ std::random_device{}() };
		static std::uniform_int_distribution<int> u;
		return u(e, std::uniform_int_distribution<int>::param_type(a, b));
	}