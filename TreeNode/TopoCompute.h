#ifndef TOPOCOMPUTE_H
#define TOPOCOMPUTE_H
#include <fstream>
#include <map>
#include <commonlib.h>
#include <Node.h>
#include <tensor_math_lib.h>
#include <cstdarg>

namespace TopoENV{
    void ComputeAll(map<string, TopoENV::TopoNode> &indgree, MULTINode &tnode);
    void ProcessBracket(string shape_str, std::vector<int> &shape_v);
    MULTINode ConstructTree(string filepath, map<string, TopoNode> &map_s_n);
    void TopoComputeEngine(map<string, TopoNode> indgree, Eigen::MatrixXd input);
}

#endif