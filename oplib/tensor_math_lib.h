#ifndef TENSOR_MATH_LIB_H
#define TENSOR_MATH_LIB_H

#include <exception>
#include <cmath>
#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

namespace oplib {
	template<typename T>
	T sigmoid(T x){
		return T(1.0)/(T(1.0) + exp(-x));
	}

	template<typename T>
	T relu(T x){
		return x < T(0.0) ? T(0.0) : x;
	}
	Eigen::MatrixXd AddBroadCast(MatrixXd mat1, MatrixXd mat2, bool cond);
}

#endif
