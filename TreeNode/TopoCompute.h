#ifndef TOPOCOMPUTE_H
#define TOPOCOMPUTE_H
#include <fstream>
#include <map>
#include <commonlib.h>
#include <DataType/datatype.h>
#include <Node.h>
#include <cstdarg>

namespace TopoENV{
    void ProcessBracket(string shape_str, std::vector<int> &shape_v){
	    string inner_str = shape_str.substr(1, shape_str.size()-2); //leave out the left and right bracket
	    inner_str.append(",");
	    int start = 0;
	    for (int i = 0; i < inner_str.size(); i ++){
		    auto ch = inner_str[i];
		    if (ch == ','){
			    string num = inner_str.substr(start, i - start);
			    if (num == "?"){
				    shape_v.push_back(-1);
			    }else{
				    shape_v.push_back(stoi(num));
			    }
			    start = i + 1; // skip the possible blank
			    i ++; // skip to the next number
		    }
	    }//end for
        if (shape_v.size() == 1){
            shape_v.push_back(1);
        }
    }

    template<typename T>
    MULTINode<T> ConstructTree(string filepath, map<string, TopoNode<T>> &map_s_n){
	    JsonReader reader;
	    JsonValue froot;
	    string colon = ":";
	    MULTINode<T> root;
	    ifstream in(filepath, ios::binary);
	    if (!in.is_open()){
		    cout << "Error opening file\n";
		    return root;
	    }
	    if (reader.parse(in, froot)){
		    for (int i = 0; i < froot.size(); i ++){
			    string step_name = "step";
				step_name.append(to_string(i));
			    JsonValue sub_root = froot[step_name]; 
                cout << "#####" << step_name << "#####" << endl;
			    JsonValue::Members op_name = sub_root.getMemberNames(); //for each node name in stepi
			    for (auto iter = op_name.begin(); iter != op_name.end(); iter ++){
				    JsonValue node = sub_root[*iter];
				    string new_node_name = step_name;
				    new_node_name.append(colon);
				    new_node_name.append(*iter);
                    cout << "*****" << new_node_name << "*****" << endl;
				    int row_size = -2;
				    int col_size = -2;
				    std::vector<int> shape;
				    if (!node["SHAPE"].isNull()){
					    ProcessBracket(node["SHAPE"].asCString(), shape);
					    row_size = shape[0];
                        col_size = shape[1];
				    }
				    Eigen_2D<T> Temp_data;
                    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& matr_data = Temp_data.getData();
				    if (!node["TENSOR_VALUE"].isNull()){
					    matr_data.resize(row_size, col_size);
                        if (col_size == 1){
                            //in case tensor is one vector, not a matrix
                            for (int i = 0; i < row_size; i ++){
                                matr_data(i, 0) = node["TENSOR_VALUE"][i].asDouble();
                            }
                        }else{
                            for (int i = 0; i < row_size; i ++){
						        for (int j = 0; j < col_size; j ++){
							        matr_data(i, j) = node["TENSOR_VALUE"][i][j].asDouble();
						        }
					        }
                        }//end else
				    }
				    std::vector<string> INPUT;
				    for (int i = 0; i < node["INPUT"].size(); i ++){
					    INPUT.push_back(node["INPUT"][i].asCString());
				    }//end for
				    string op = node["OPERATION"].asCString();
				    string type = node["TYPE"].asCString();
                    string father_name = node["FATHER"].isNull()? "None" : node["FATHER"].asCString();
				    MULTINode<T> newnode(Temp_data, father_name, shape, new_node_name, op, INPUT);
                    TopoNode<T> topo_node(newnode, INPUT.size());
				    root.reset(newnode);
				    map_s_n.insert(make_pair(new_node_name, topo_node));
			    }
		    }//end for
	    }
	    return root;
    }

    template<typename T>
	void ComputeAll(map<string, TopoNode<T>> &indgree, MULTINode<T> &tnode){
        std::map<string, int> namemap;
        namemap.insert(make_pair("Add", 1));
        namemap.insert(make_pair("MatMul", 2));
        namemap.insert(make_pair("Softmax", 3));
        Eigen_2D<T> input_1;
        Eigen_2D<T> input_2;
        Eigen_2D<T> output;
        std::vector<int> new_shape;
        Eigen_Vector<T> final_res;
        switch(namemap[tnode.op]){
            case 1:
                input_1 = indgree[tnode.children[0]].getNode().data;
                input_2 = indgree[tnode.children[1]].getNode().data;
                if (input_1.get_col_length() != input_2.get_col_length() || input_1.get_row_length() != input_2.get_row_length()){
                    output = input_1.AddBoradCast(input_2);
                }else{
                    output = input_1.AddWithoutBroadCast(input_2);
                }
                new_shape.push_back(output.get_row_length());
                new_shape.push_back(output.get_col_length());
                tnode.setData(output);
                tnode.setShape(new_shape);
                break;
            case 2:
                input_1 = indgree[tnode.children[0]].getNode().data;
                input_2 = indgree[tnode.children[1]].getNode().data;
                output = input_1.Matmul(input_2);
                new_shape.push_back(output.get_row_length());
                new_shape.push_back(output.get_col_length());
                tnode.setData(output);
                tnode.setShape(new_shape);
                break;
            case 3:
                input_1 = indgree[tnode.children[0]].getNode().data;
                output = input_1.softmax2d(true);
                new_shape.push_back(output.get_row_length());
                new_shape.push_back(output.get_col_length());
                tnode.setData(output);
                tnode.setShape(new_shape);
                final_res = output.Argmax(true);
                cout << "Final Result:" << endl;
                final_res.Printout();
                break;
            default:
                cout << "Invalid Operation!!!" << endl;
        }
    }

    template<typename T>
    void TopoComputeEngine(map<string, TopoNode<T>> indgree, Eigen_2D<T> const input){
        std::deque<string> q;
        //initialzied the queue with node of which the indegree in 0
        for (auto iter = indgree.begin(); iter != indgree.end(); iter ++){
           if (iter->second.getDegree() == 0){
               //store the value of iter
               q.push_back(iter->first);
           }
        }//end for
        while(q.size() > 0){
             int size_of_q = q.size();
             while (size_of_q > 0){
                 string nodename = q.front();
                 q.pop_front();
                 MULTINode<T>& tnode = indgree[nodename].getNode();
                 //update indgree and queue
                 if (tnode.father != "None"){
                    indgree[tnode.father].setDegree(indgree[tnode.father].getDegree() - 1);
                    if (indgree[tnode.father].getDegree() == 0){
                        q.push_back(tnode.father);
                    }
                 }
                 string op_name = tnode.op;
                 if (op_name == "Placeholder"){
                     tnode.setData(input);
                 }else if(op_name == "Identity"){
                    // cout << tnode.data.getData() << endl;
                 }else{
                     ComputeAll<T>(indgree, tnode);
                 }
                 size_of_q --;
             }
        }//end for outer while
    }
}

#endif