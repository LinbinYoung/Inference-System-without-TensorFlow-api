#!/bin/bash
# Bean TensorFlow

clear

echo "Make the Project and Run Test"

make clean

make

cd build

./output

cd ../

