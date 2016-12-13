#!/bin/bash

THEANO_FLAGS="device=gpu0" python2.7 mnist.py -c config_fullurnn_128.yaml

THEANO_FLAGS="device=gpu0" python2.7 mnist.py -c config_lstm_40.yaml

THEANO_FLAGS="device=gpu0" python2.7 mnist.py -c config_resurnn_128.yaml
