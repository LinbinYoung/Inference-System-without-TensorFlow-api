#ifndef NODE_H
#define NODE_H
#include <commonlib.h>
#include <vector>
#include <string>
#include <algorithm>
#include <DataType/datatype.h>
#include <DataType/MultiData.h>

using namespace MultiEigen;

namespace TopoENV{
    /*
        1. MULTINode: Multi-way Tree structure, used to store Tensor Node
        2. TopoNode: MULTINode + indegree
    */
    template <typename T>
    class MULTINode{
        public:
            TensorData<T> data;
            string father;
            std::vector<int> shape;
            std::vector<int> stride;
            std::vector<string> children;
            std::string name;
            std::string op;
            MultiEigen::padding_type pad_type;
        public:
            MULTINode(){}
            ~MULTINode(){}
            MULTINode(TensorData<T> &new_data, string new_father, std::vector<int> new_shape, std::vector<int> new_stride, std::string new_name, std::string new_op, std::vector<string> child_list, int indegree, MultiEigen::padding_type pad_type){
                // Leaf Node like Variable and Placeholder
                this->data = new_data;
                this->father = new_father;
                this->shape.assign(new_shape.begin(), new_shape.end());
                this->stride.assign(new_stride.begin(), new_stride.end());
                this->name = new_name;
                this->op = new_op;
                this->children.assign(child_list.begin(), child_list.end());
                this->indegree = indegree;
                this->pad_type = pad_type;
            }
            void reset (const MULTINode<T> &instan){
                this->data = instan.data;
                this->father = instan.father;
                this->shape.assign(instan.shape.begin(), instan.shape.end());
                this->stride.assign(instan.stride.begin(), instan.stride.end());
                this->name = instan.name;
                this->op = instan.op;
                this->children.assign(instan.children.begin(), instan.children.end());
                this->indegree = instan.getDegree();
                this->pad_type = instan.pad_type;
            }
            void setData(TensorData<T> const &new_data){
                this->data = new_data;
            }
            std::string getName(){
                return this->father;
            }
            std::vector<int>& getShape(){
                return this->shape;
            }
            void setShape(std::vector<int> new_shape){
                this->shape.assign(new_shape.begin(), new_shape.end());
            }
            void setStride(std::vector<int> new_stride){
                this->stride.assign(new_stride.begin(), new_stride.end());
            }
            int getDegree() const{
                return this->indegree;
            }
            void setDegree(int val){
                this->indegree = val;
            }
        protected:
            int indegree;
    };
}

#endif