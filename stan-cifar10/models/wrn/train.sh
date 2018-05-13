#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
	--solver=models/wrn/solver.prototxt --gpu=0$@ 
