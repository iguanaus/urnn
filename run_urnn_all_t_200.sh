#!/bin/bash
#This is the first file to run on myCal.
niter=10000
batch_size=128
learning_rate=0.001
n_cats=8

mkdir -p exp

for t in 200; do
    for model in LSTM,40,adhoc complex_RNN,128,full complex_RNN,64,adhoc ; do IFS=","; set $model
        SECONDS=0
        w_impl=$3
        echo "Running memory_problem experiment for N=$2 $1 with time_steps=$t"
        cmd="THEANO_FLAGS='device=gpu0' python2.7 -u memory_problem.py --n_iter $niter --n_batch ${batch_size} --n_hidden $2 --time_steps $t --learning_rate 0.001 --savefile exp/memory_problem_$1_$3_$1_nhidden$2_t$t --model $1 --input_type categorical --out_every_t True --loss_function CE --w_impl $w_impl --num_cats $n_cats"
        echo $cmd
        eval $cmd
        echo "Experiment took $SECONDS seconds."
    done
done