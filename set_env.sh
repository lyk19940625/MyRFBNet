#!/usr/bin/env bash
echo "[global]">>/root/.pip/pip.conf
echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple">>/root/.pip/pip.conf
echo "export CUDAHOME=/usr/local/cuda">>/root/.bashrc
source /root/.bashrc
