#ifndef TOPOCOMPUTE_H
#define TOPOCOMPUTE_H
#include <json/json.h>
#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <Node.h>
#include <tensor_math_lib.h>

using namespace Json;
using namespace Eigen;
using namespace oplib;

namespace TopoENV{

    struct TopoNode{
        //overload construct function for initailization
        public:
            TopoNode(){}
            TopoNode(MULTINode newnode, int in){
                //copy the value of newnode
                this->node = newnode;
                this->indegree = in;
            }
            MULTINode& getNode(){
                //why we should put one & here
                //We want to transfer the reference of the varibale 
                return this->node;
            }
            int getDegree() const{
                return this->indegree;
            }
            void setDegree(int val){
                this->indegree = val;
            }
        protected:
            //less strict than private, could be adopted by its derivatives
            MULTINode node;
            int indegree;
    };

    void ProcessBracket(string shape_str, std::vector<int> &shape_v);
    MULTINode ConstructTree(string filepath, map<string, TopoNode> &map_s_n);
    void TopoComputeEngine(map<string, TopoNode> indgree, Eigen::MatrixXd input);
    // void ConstructIndgree(MULTINode root, map<string, TopoNode> &indgree);
}

#endif