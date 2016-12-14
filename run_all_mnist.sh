#!/bin/bash
#This is the mnist 1

THEANO_FLAGS="device=gpu0" python2.7 mnist.py -c config_fullurnn_256.yaml

