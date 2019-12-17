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
  string path = "../jsonfile/test.json";
  std::map<string, TopoENV::TopoNode<double>> indgree;
  TopoENV::ConstructTree<double>(path, indgree);
  Eigen::MatrixXd m = MatrixXd::Random(32,784);
  Eigen_2D<double> input;
  input.setData(m);
  TopoENV::TopoComputeEngine<double>(indgree, input);
	return 0;
}