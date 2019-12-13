#ifndef NODE_H
#define NODE_H
#include <commonlib.h>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

namespace TopoENV{
    class MULTINode{
        public:
            Eigen::MatrixXd data;
            std::string father;
            std::vector<int> shape;
            std::vector<string> children;
            std::string name;
            std::string op;
        public:
            MULTINode(){}
            ~MULTINode(){}
            MULTINode(Eigen::MatrixXd &new_data, std::string new_father, std::vector<int> new_shape, std::string new_name, std::string new_op, std::vector<string> child_list){
                // Leaf Node like Variable and Placeholder
                this->data = new_data;
                this->father = new_father;
                this->shape.assign(new_shape.begin(), new_shape.end());
                this->name = new_name;
                this->op = new_op;
                this->children.assign(child_list.begin(), child_list.end());
            }
            void reset (const MULTINode& instan){
                this->data = instan.data;
                this->father = instan.father;
                this->shape.assign(instan.shape.begin(), instan.shape.end());
                this->name = instan.name;
                this->op = instan.op;
                this->children.assign(instan.children.begin(), instan.children.end());
            }
            Eigen::MatrixXd& getData(){
                return this->data;
            }
            void setData(Eigen::MatrixXd const &new_data){
                this->data = new_data;
            }
            std::string getName(){
                return this->father;
            }
            std::vector<int> getShape(){
                return this->shape;
            }
            void setShape(std::vector<int> new_shape){
                this->shape.assign(new_shape.begin(), new_shape.end());
            }
    };
    struct TopoNode{
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
}

#endif