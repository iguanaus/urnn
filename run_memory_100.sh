niter=10000
batch_size=20
n_hidden=64
time_steps=100
learning_rate=0.001
savefile='exp/memory_problem_complex_RNN_full_complex_RNN_nhidden128_t1000'
model='complex_RNN'
input_type='categorical'
out_every_t='True'
loss_function='CE'
w_impl='full'

cmd="THEANO_FLAGS='device=gpu0' python2.7 -u memory_problem.py $niter ${batch_size} ${n_hidden} ${time_steps} ${learning_rate} ${savefile} $model ${input_type} ${out_every_t} ${loss_function} ${w_impl}"
#cmd="THEANO_FLAGS='device=gpu0' python2.7 -u memory_problem.py 10000 20 64 100 0.001 exp/memory_problem_complex_RNN_full_complex_RNN_nhidden128_t1000 complex_RNN categorical True CE full"
mkdir -p exp

echo $cmd
eval $cmd