cmake_minimum_required(VERSION 3.14)
project(C_inference)

set(CMAKE_CXX_STANDARD 11)

link_directories(./jsoncpp/lib/)

link_libraries(libjsoncpp.a)

include_directories(./jsoncpp/include ./Eigen ./)

add_executable(C_inference main.cpp common/commonlib.h DataType/datatype.h DataType/MultiData.h TreeNode/activator.h TreeNode/Node.h TreeNode/TopoCompute.h LoadBatch/loaddata.h)