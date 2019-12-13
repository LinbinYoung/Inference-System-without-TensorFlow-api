#include <tensor_math_lib.h>

namespace TopoENV {
    Eigen::MatrixXd AddBroadCast(MatrixXd mat1, MatrixXd mat2, bool cond){
        /*
            mat1: The first Matrix
            mat2: The second Matrix (could be a vector)
            cond: broadcast or not
        */
       Eigen::MatrixXd output;
       if (!cond){
           mat1 = mat1 + mat2;
       }else{
           Eigen::VectorXd v(mat2.rows());
           for(int i = 0; i < mat2.rows(); i ++){
               v(i) = mat2(i,0);
           }
           output = mat1.transpose();
           cout << output.rows() << " : " << output.cols() << endl;
           output.colwise() += v;
           return output.transpose();
       }
       return output;
    }

    Eigen::MatrixXd SoftmaxOp(MatrixXd mat1, bool row_or_col){
        /*
            True: Softmax According to Row
            False: Softmax According to Col
        */
        if (!row_or_col){
            return SoftmaxOp(mat1.transpose(), !row_or_col).transpose();
        }
        for (int i = 0; i < mat1.rows(); i ++){
            MySoftmax(mat1, i, mat1.cols());
        }
        return mat1;
    }

    void MySoftmax(MatrixXd &mat, int m, int n){
        double sum = 0.0;
        double max = FindMax(mat, m, n);
        for (int i = 0; i < n; i ++){
            double fi = exp(mat(m, i) - max);
            mat(m, i) = fi;
            sum = sum + fi;
        }
        for (int i = 0; i < n; i ++){
            mat(m, i) = mat(m, i) / sum;
        }
    }

    Eigen::VectorXi ArgmaxOp(MatrixXd mat1, bool row_or_col){
        if (!row_or_col){
            return ArgmaxOp(mat1.transpose(), !row_or_col);
        }
        Eigen::VectorXi output;
        output.resize(mat1.rows());
        for (int i = 0; i < mat1.rows(); i ++){
            output(i) = FindMaxIndex(mat1, i, mat1.cols());
        }
        return output;
    }

    double FindMax(MatrixXd& mat1, int m, int n){
        double max = numeric_limits<double>::min();
        for (int i = 0; i < n; i ++){
            if (max < mat1(m, i)) max = mat1(m, i);
        }//end for
        return max;
    }

    int FindMaxIndex(MatrixXd& mat1, int m, int n){
        double max = numeric_limits<double>::min();
        int res = 0;
        for (int i = 0; i < n; i ++){
            if (max < mat1(m, i)) {
                max = mat1(m, i);
                res = i;
            }
        }//end for
        return res;
    }
}