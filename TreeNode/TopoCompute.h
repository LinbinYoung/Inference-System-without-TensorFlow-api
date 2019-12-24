#ifndef TOPOCOMPUTE_H
#define TOPOCOMPUTE_H
#include <fstream>
#include <map>
#include <commonlib.h>
#include <TreeNode/activator.h>
#include <Node.h>
#include <cstdarg>

using namespace MATHLIB;

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
    MULTINode<T> ConstructTree(string filepath, map<string, MULTINode<T>>& map_s_n){
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
                    string op = node["OPERATION"].asCString();
				    int dimension = 1; // default dimension is 1
				    std::vector<int> shape;
				    if (!node["SHAPE"].isNull()){
                        cout << node["SHAPE"].asCString() << endl;
					    ProcessBracket(node["SHAPE"].asCString(), shape);
					    dimension = shape.size();
				    }
                    cout << dimension << endl;
                    TensorData<T> temp_data(tyname::D_0);
                    if (dimension == 2){
                        if (shape[1] == 1){
                            temp_data.setType(tyname::D_1);
                            Eigen_Vector<T>& s_data = temp_data.E1D;
                            if (!node["TENSOR_VALUE"].isNull()){s_data.setData(shape[0], node["TENSOR_VALUE"]);}
                        }else{
                            temp_data.setType(tyname::D_2);
                            Eigen_2D<T>& s_data = temp_data.E2D;
                            if (!node["TENSOR_VALUE"].isNull()){s_data.setData(shape[0], shape[1], node["TENSOR_VALUE"]);}
                        }
                    }else if (dimension == 3){
                        temp_data.setType(tyname::D_3);
                        Eigen_3D<T>& s_data = temp_data.E3D;
                        if (!node["TENSOR_VALUE"].isNull()){s_data.setData(shape[0], shape[1], shape[2], node["TENSOR_VALUE"]);}
                    }else if (dimension == 4){
                        temp_data.setType(tyname::D_4);
                        cout << (temp_data.getType() == tyname::D_4) << endl;
                        //cout << temp_data.getType() << endl;
                        Eigen_4D<T>& s_data = temp_data.E4D;
                        if (op == "Identity"){
                            //which means this is kernel matrix
                            if (!node["TENSOR_VALUE"].isNull()){s_data.setData(shape[0], shape[1], shape[2], shape[3], node["TENSOR_VALUE"], matrix_type::kernel);}
                            cout << "Warm Hug" << endl;
                        }
                    }
				    std::vector<string> INPUT;
				    for (int i = 0; i < node["INPUT"].size(); i ++){
					    INPUT.push_back(node["INPUT"][i].asCString());
				    }//end for
				    string type = node["TYPE"].asCString();
                    string father_name = node["FATHER"].isNull()? "None" : node["FATHER"].asCString();
				    MULTINode<T> newnode(temp_data, father_name, shape, *iter, op, INPUT, INPUT.size());
				    root.reset(newnode);
                    //store the address of this object
				    map_s_n.insert(make_pair(*iter, newnode));
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
        namemap.insert(make_pair("Reshape", 4));
        namemap.insert(make_pair("Sigmoid", 5));
        namemap.insert(make_pair("Relu", 6));
        Eigen_2D<T> input_1;
        Eigen_2D<T> input_2;
        Eigen_4D<T> image_input;
        Eigen_2D<T> output;
        std::vector<int> new_shape;
        Eigen_Vector<T> final_res;
        switch(namemap[tnode.op]){
            case 1:
                cout << "1" << endl;
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
                cout << "1 finished" << endl;
                break;
            case 2:
                cout << "2" << endl;
                input_1 = indgree[tnode.children[0]].getNode().data;
                cout << input_1.get_col_length() << "=====" << input_1.get_row_length() << endl;
                input_2 = indgree[tnode.children[1]].getNode().data;
                cout << input_2.get_col_length() << "=====" << input_2.get_row_length() << endl;
                output = input_1.Matmul(input_2);
                new_shape.push_back(output.get_row_length());
                new_shape.push_back(output.get_col_length());
                tnode.setData(output);
                tnode.setShape(new_shape);
                cout << "2 finished" << endl;
                break;
            case 3:
                cout << "3" << endl;
                input_1 = indgree[tnode.children[0]].getNode().data;
                output = input_1.softmax2d(true);
                new_shape.push_back(output.get_row_length());
                new_shape.push_back(output.get_col_length());
                tnode.setData(output);
                tnode.setShape(new_shape);
                final_res = output.Argmax(true);
                cout << "Final Result:" << endl;
                final_res.Printout();
                cout << "3 finished" << endl;
                break;
            case 4:
                /*
                [a, b, c, d]
                -a: Number of pictures 
                -b: x dimension
                -c: y dimension
                -d: Number of channels
                */
                break;
            case 5:
                cout << "5" << endl;
                input_1 = indgree[tnode.children[0]].getNode().data;
                output = input_1.apply(MATHLIB::sigmoid<T>);
                new_shape.push_back(output.get_row_length());
                new_shape.push_back(output.get_col_length());
                tnode.setData(output);
                tnode.setShape(new_shape);
                cout << "5 finished" << endl;
                break;
            case 6:
                cout << "6" << endl;
                input_1 = indgree[tnode.children[0]].getNode().data;
                output = input_1.apply(MATHLIB::relu<T>);
                new_shape.push_back(output.get_row_length());
                new_shape.push_back(output.get_col_length());
                tnode.setData(output);
                tnode.setShape(new_shape);
                cout << "6 finished" << endl;
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