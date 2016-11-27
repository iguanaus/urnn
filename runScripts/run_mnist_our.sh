#!/bin/bash

THEANO_FLAGS="device=gpu0" python2.7 mnist.py -c config_mnist_fulluRNN.yaml
