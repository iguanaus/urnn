#This will generate the full-capac URNN for T=100,200,and 500
for tval in 1000
do
niter=10000
batch_size=128 
n_hidden=40
time_steps=$tval
learning_rate=0.001
num_cats=5
savefile='exp/memory_problem_full_complex_RNN_learning_0.001_ncats_'$num_cats'_nhidden40_t'$time_steps
model='LSTM'
input_type='categorical'
out_every_t='True'
loss_function='CE'
w_impl='adhoc'

cmd="THEANO_FLAGS='device=gpu0' python2.7 -u memory_problem.py $niter ${batch_size} ${n_hidden} ${time_steps} ${learning_rate} ${savefile} $model ${input_type} ${out_every_t} ${loss_function} ${w_impl} ${num_cats}"
#cmd="THEANO_FLAGS='device=gpu0' python2.7 -u memory_problem.py 10000 20 64 100 0.001 exp/memory_problem_complex_RNN_full_complex_RNN_nhidden128_t1000 complex_RNN categorical True CE full"
mkdir -p exp

echo $cmd
eval $cmd
done

