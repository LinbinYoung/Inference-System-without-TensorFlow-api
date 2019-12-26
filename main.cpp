#include <iostream>
#include <map>
#include <common/commonlib.h>
#include <TreeNode/Node.h>
#include <TreeNode/TopoCompute.h>

using namespace std;
using namespace TopoENV;

/*

enum ValueType {
  nullValue = 0, ///< 'null' value
  intValue,      ///< signed integer value
  uintValue,     ///< unsigned integer value
  realValue,     ///< double value
  stringValue,   ///< UTF-8 string value
  booleanValue,  ///< bool value
  arrayValue,    ///< array value (ordered list)
  objectValue    ///< object value (collection of name/value pairs).
};

*/

int main(void){
  string path = "../jsonfile/model2.json";
  std::map<string, MULTINode<double>> indgree;
  TopoENV::ConstructTree<double>(path, indgree);
  // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m =  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Random(32,784);
  // Eigen_2D<double> s_input;
  // s_input.setData(m);
  // input.setData(s_input);
  // Eigen_4D<double> temp(32,28,28,1);
  Eigen_2D<double> temp(32,784);
  TensorData<double> input;
  input.setData(temp);
  TopoENV::TopoComputeEngine<double>(indgree, input);
  // Eigen_4D<double> kernel;
  // kernel.setData(28,28,1,5,matrix_type::kernel);
  // Eigen_2D<double> res = temp.reshape();
  // cout << res.getData() << endl;
	return 0;
}