#include <iostream>
#include <map>
#include <common/commonlib.h>
#include <TreeNode/Node.h>
#include <TreeNode/TopoCompute.h>
#include <LoadBatch/loaddata.h>

using namespace std;
using namespace TopoENV;
using namespace DataLoader;

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
    string model_path = "../jsonfile/Convolution_Model.json";
    string file_path = "../jsonfile/data15.json";
    TensorData<double> input;
    Eigen_4D<double> res = DataLoader::loadDataset_4D<double>(file_path);
    input.setData(res);
    std::map<string, MULTINode<double>> indgree;
    TopoENV::ConstructTree<double>(model_path, indgree);
    TopoENV::TopoComputeEngine<double>(indgree, input);
    return 0;
}

// data
// TensorFlow: [1 7 2 7 9 1 3 7 4 6 0 9 1 8 3 1 1 7 8 3 8 3 6 0 2 3 0 1 0 2 0 9]
// Me: [1 7 2 7 8 1 3 7 4 6 0 9 1 8 3 1 1 7 8 3 8 3 6 0 2 3 0 1 0 2 0 9]

// data1
// TensorFlow: [5 0 2 4 1 6 1 4 0 6 6 7 6 1 0 8 3 7 2 1 0 3 2 8 1 5 8 2 8 0 3 1]
// Me: [5 0 2 4 1 6 1 4 0 6 6 7 6 1 0 8 3 7 2 1 0 3 2 8 1 5 8 2 8 0 3 1]

// data2
// TensorFlow: [6 9 4 8 4 6 4 0 0 1 9 6 5 1 3 2 4 4 2 0 3 9 4 6 8 5 3 2 7 6 1 1]
// Me: [6 9 4 8 4 6 4 0 0 1 9 6 5 1 3 2 4 4 2 0 3 9 4 6 8 5 3 2 7 6 1 1 ]

// data3
// TensorFlow: [0 8 7 9 9 8 3 0 8 0 5 4 3 3 5 2 6 4 9 9 2 8 2 0 0 0 8 6 2 7 2 0]
// Me: [0 8 7 9 9 8 3 0 8 0 5 4 3 3 5 2 6 4 9 9 2 8 2 0 0 0 8 6 2 7 2 0 ]

// data15
// TensorFlow: [0 6 2 3 1 8 5 4 6 4 7 5 9 3 1 7 0 7 9 2 0 5 2 7 8 9 5 3 0 9 7 8]
// Me: [0 6 2 3 1 8 5 4 6 4 7 5 9 3 1 7 0 7 9 2 0 5 2 7 8 9 5 3 0 9 7 8 ]