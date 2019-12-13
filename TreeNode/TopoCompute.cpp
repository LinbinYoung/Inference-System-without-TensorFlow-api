#include <TopoCompute.h>

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

    MULTINode ConstructTree(string filepath, map<string, TopoNode> &map_s_n){
	    //load the json into memeory
	    JsonReader reader;
	    JsonValue froot;
	    string colon = ":";
	    TopoENV::MULTINode root; //It would automatically call the default constructor
	    ifstream in(filepath, ios::binary);
	    if (!in.is_open()){
		    cout << "Error opening file\n";
		    return root;
	    }
	    if (reader.parse(in, froot)){
		    // cout << "Total Number of Steps are " << froot.size() << endl;
		    for (int i = 0; i < froot.size(); i ++){
			    string step_name = "step";
				step_name.append(to_string(i));
			    JsonValue sub_root = froot[step_name]; 
			    JsonValue::Members op_name = sub_root.getMemberNames(); //for each node name
			    for (auto iter = op_name.begin(); iter != op_name.end(); iter ++){
				    JsonValue node = sub_root[*iter];
				    string new_node_name = step_name;
				    new_node_name.append(colon);
				    new_node_name.append(*iter);
				    // cout << new_node_name << endl;
				    /*
					    INPUT
					    OPERATION
					    SHPAE (We now assume that the dimension of the matrix is always 2)
					    TYPE
					    TENSOR_VALUE
				    */
				    int row_size = -2;
				    int col_size = -2;
				    std::vector<int> shape;
				    if (!node["SHAPE"].isNull()){
					    ProcessBracket(node["SHAPE"].asCString(), shape);
					    row_size = shape[0];
                        col_size = shape[1];
				    }
				    Eigen::MatrixXd T_data;
				    if (!node["TENSOR_VALUE"].isNull()){
					    T_data.resize(row_size, col_size);
                        if (col_size == 1){
                            //in case tensor is one vector, not a matrix
                            for (int i = 0; i < row_size; i ++){
                                T_data(i, 0) = node["TENSOR_VALUE"][i].asDouble();
                            }
                        }else{
                            for (int i = 0; i < row_size; i ++){
						        for (int j = 0; j < col_size; j ++){
							        T_data(i, j) = node["TENSOR_VALUE"][i][j].asDouble();
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
				    MULTINode newnode(T_data, father_name, shape, new_node_name, op, INPUT);
                    TopoNode topo_node(newnode, INPUT.size());
				    root.reset(newnode);
				    map_s_n.insert(make_pair(new_node_name, topo_node));
			    }
		    }//end for
	    }
	    return root;
    }

	void ComputeAll(map<string, TopoENV::TopoNode> &indgree, MULTINode &tnode){
        std::map<string, int> namemap;
        namemap.insert(make_pair("Add", 1));
        namemap.insert(make_pair("MatMul", 2));
        namemap.insert(make_pair("Softmax", 3));
        Eigen::MatrixXd input_1;
        Eigen::MatrixXd input_2;
        Eigen::MatrixXd output;
        std::vector<int> new_shape;
        Eigen::VectorXi final_res;
        switch(namemap[tnode.op]){
            case 1:
                input_1 = indgree[tnode.children[0]].getNode().data;
                input_2 = indgree[tnode.children[1]].getNode().data;
                if (input_1.cols() != input_2.cols() || input_1.rows() != input_2.rows()){
                    output = AddBroadCast(input_1, input_2, true);
                }else{
                    output = AddBroadCast(input_1, input_2, false);
                }
                new_shape.push_back(output.rows());
                new_shape.push_back(output.cols());
                tnode.setData(output);
                tnode.setShape(new_shape);
                break;
            case 2:
                input_1 = indgree[tnode.children[0]].getNode().data;
                input_2 = indgree[tnode.children[1]].getNode().data;
                output = input_1*input_2;
                new_shape.push_back(output.rows());
                new_shape.push_back(output.cols());
                tnode.setData(output);
                tnode.setShape(new_shape);
                break;
            case 3:
                input_1 = indgree[tnode.children[0]].getNode().data;
                output = SoftmaxOp(input_1, true);
                new_shape.push_back(output.rows());
                new_shape.push_back(output.cols());
                tnode.setData(output);
                tnode.setShape(new_shape);
                final_res = ArgmaxOp(output, true);
                cout << "Final Result:" << endl;
                cout << final_res << endl;
                break;
            default:
                cout << "Invalid Operation!!!" << endl;
        }
    }

    void TopoComputeEngine(map<string, TopoNode> indgree, Eigen::MatrixXd const input){
        //copy value
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
                 MULTINode& tnode = indgree[nodename].getNode();
                 //update indgree and queue
                 if (tnode.father != "None"){
                    indgree[tnode.father].setDegree(indgree[tnode.father].getDegree() - 1);
                    if (indgree[tnode.father].getDegree() == 0){
                         //append new node to the queue
                        q.push_back(tnode.father);
                    }
                 }
                 //do operation on this node
                 string op_name = tnode.op;
                 if (op_name == "Placeholder"){
                     //initialized the tensor value of the node
                     tnode.setData(input);
                    //  cout << indgree[nodename].getNode().data << endl;
                 }else if(op_name == "Identity"){
                    //  cout << tnode.data << endl;
                 }else{
                     ComputeAll(indgree, tnode);
                 }
                 size_of_q --;
             }//end inner while
        }
    }
}