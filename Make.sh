#!/usr/bin/env bash

cd caffe-1.0
echo "MAKE CAFFE"
make clean && make all -j 50
make distribute
echo "MAKE COMMON"
cd ../common
make clean && make
echo "MAKE SERVICE"
cd ../service
make clean && make -j 2
echo "MAKE TONIC CLIENT"
cd ../tonic-suite/img -j 4
make clean && make
