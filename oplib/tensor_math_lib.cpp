#include <tensor_math_lib.h>

namespace oplib {
    Eigen::MatrixXd AddBroadCast(MatrixXd mat1, MatrixXd mat2, bool cond){
        /*
            mat1: The first Matrix
            mat2: The second Matrix (could be a vector)
            cond: broadcast or not
        */
       MatrixXd output;
       if (!cond){
           output = mat1 + mat2;
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
}