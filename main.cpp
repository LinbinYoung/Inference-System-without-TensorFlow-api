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
  string path = "../jsonfile/model3.json";
  std::map<string, MULTINode<double>> indgree;
  TopoENV::ConstructTree<double>(path, indgree);
  //Variable_4/read
  string name = "Variable_3/read";
  MULTINode<double> temp = indgree[name];
  cout << temp.getDegree() << endl;
  cout << "LINBIN"<< endl;
  cout << temp.op << endl;
  cout << temp.name << endl;
  cout << (temp.data.getType() == tyname::D_4) << endl;
  cout << temp.data.E4D.getData().size() << endl;
  // if (temp->data.getType() == tyname::D_4){
  //   cout << temp->data.E4D.reshape().getData() << endl;
  // }
  // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m =  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Random(32,784);
  // Eigen_2D<double> input;
  // input.setData(m);
  // TopoENV::TopoComputeEngine<double>(indgree, input);
  // Eigen_4D<double> temp(1,2,3,23);
  // Eigen_2D<double> res = temp.reshape();
  // cout << res.getData() << endl;
	return 0;
}