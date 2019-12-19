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
  std::map<string, TopoENV::TopoNode<double>> indgree;
  TopoENV::ConstructTree<double>(path, indgree);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m =  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Random(3,3);
  Eigen_2D<double> input;
  input.setData(m);
  // iter ++;
  for (auto iter = input.begin(); iter != input.end(); iter++){
     for (auto inner_iter = iter.begin(); inner_iter != iter.end(); inner_iter++){
       cout << *inner_iter << " ";
     }
     cout << endl;
  }
  Eigen_2D<double> output = input.apply(MATHLIB::sigmoid<double>);
  cout << output.getData() << endl;
  //TopoENV::TopoComputeEngine<double>(indgree, input);
	return 0;
}