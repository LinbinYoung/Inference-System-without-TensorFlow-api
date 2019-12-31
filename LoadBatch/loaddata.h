//
// Created by Linbin Yang on 2019-12-30.
//

#ifndef C_INFERENCE_LOADDATA_H
#define C_INFERENCE_LOADDATA_H
#include <fstream>
#include <common/commonlib.h>
#include <DataType/datatype.h>

using namespace MultiEigen;

namespace DataLoader{
    template <typename T>
    Eigen_4D<T> loadDataset_4D(string filepath){
        JsonReader reader;
        JsonValue froot;
        Eigen_4D<T> res;
        ifstream in(filepath, ios::binary);
        if (!in.is_open()){
            cout << "Error opening fiel\n";
            return res;
        }
        if (reader.parse(in, froot)){
             string key = "first";
             JsonValue v =  froot[key];
             res.setData(32,28,28,1,v,matrix_type::image);
        }
        return res;
    }

    template <typename T>
    Eigen_2D<T> loadDataset_2D(string filepath, std::vector<int> shape){
        JsonReader reader;
        JsonValue froot;
        Eigen_2D<T> res;
        ifstream in(filepath, ios::binary);
        if (!in.is_open()){
            cout << "Error opening fiel\n";
            return res;
        }
        if (reader.parse(in, froot)){
            string key = "first";
            JsonValue v =  froot[key];
            res.setData(shape[0],shape[1],v);
        }
        return res;
    }
}

#endif //C_INFERENCE_LOADDATA_H
