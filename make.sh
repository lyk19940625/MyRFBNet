#!/usr/bin/env bash
cd ./utils/

rm -rf ./build/
cd ./nms/
rm -rf *.so
cd ../pycocotools/
rm -rf *.so
cd ../

CUDA_PATH=/usr/local/cuda-8.0/

python build.py build_ext --inplace

cd ..
