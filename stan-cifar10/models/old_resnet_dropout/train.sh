#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
	--solver=models/old_resnet/solver.prototxt $@ 
