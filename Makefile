SRCS = main.cpp ./TreeNode/TopoCompute.cpp ./TreeNode/tensor_math_lib.cpp

TARGET = output

INCLUDE = -std=c++11 -I/usr/lib -I/usr/include/ -I/usr/include/jsoncpp/include -I/usr/local/include/eigen3 -I./ -I./TreeNode -I./tensor_math_lib -I./DataType -I./common

LIB = -L/usr/lib/libjson_linux-gcc-5.4.0_libmt.so /usr/lib/libjson_linux-gcc-5.4.0_libmt.a

.PHONY:all clean

all: $(TARGET)
# name of library
$(TARGET): $(SRCS)
	mkdir build && cd build
	g++ -g $^ $(INCLUDE) $(LIB) -o $@
	mv ./$(TARGET) ./build  

clean:
	rm -rf build