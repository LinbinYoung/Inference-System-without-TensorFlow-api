#!/bin/bash
# Bean TensorFlow

echo "Make the Project and Run Test"

make clean

make

cd build

./output

cd ../

