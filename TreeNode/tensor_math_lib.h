#ifndef TENSOR_MATH_LIB_H
#define TENSOR_MATH_LIB_H

#include <commonlib.h>
#include <exception>
#include <cmath>
#include <iostream>

using namespace std;
using namespace Eigen;

namespace TopoENV {
	template<typename T>
	T sigmoid(T x){
		return T(1.0)/(T(1.0) + exp(-x));
	}

	template<typename T>
	T relu(T x){
		return x < T(0.0) ? T(0.0) : x;
	}
	Eigen::MatrixXd AddBroadCast(MatrixXd mat1, MatrixXd mat2, bool cond);
	Eigen::MatrixXd SoftmaxOp(MatrixXd mat1, bool row_or_col);
	Eigen::VectorXi ArgmaxOp(MatrixXd mat1, bool row_or_col);
	void MySoftmax(MatrixXd &mat, int m, int n);
	double FindMax(MatrixXd& mat1, int m, int n);
	int FindMaxIndex(MatrixXd& mat1, int m, int n);
}

#endif