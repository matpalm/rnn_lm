#!/usr/bin/env python

# simplest-as-we-can rnn model

# - no gating within unit at all
# - no (other) protection against exploding or vanishing gradients
# - no batching, train one example at a time.
# - trivial randn weight init
# - no bias with dot products

import sys, time, optparse
import numpy as np
import util
import reber_grammar as rb
import theano
import theano.tensor as T
from simple_rnn import SimpleRnn
from bidirectional_rnn import BidirectionalRnn
from gru_rnn import GruRnn
from attention_rnn import AttentionRnn

optparser = optparse.OptionParser(prog='rnn')
optparser.add_option('--adaptive-learning-rate', None, dest='adaptive_learning_rate_fn', type='string',
                     default="rmsprop", help='adaptive learning rate method')
optparser.add_option('--type', None, dest='type', type='string',
                     default="", help='rnn type; simple, bidirectional, gru or attention')
optparser.add_option('--epochs', None, dest='epochs', type='int',
                     default=5, help='number of epoches to run')
optparser.add_option('--num-hidden', None, dest='num_hidden', type='int',
                     default=10, help='hidden node dimensionality')
opts, _arguments = optparser.parse_args()

# data is just characters, ['A', 'B', 'A', ... ]
# but we left and right pad with <s> and </s> to include prediction of start/end of sequence
# and convert to idxs

# matrix sizing
n_in = rb.vocab_size()
n_hidden = opts.num_hidden

# t_x input and t_y output sequence
t_x = T.ivector('x')  # eg s A B A D   for sequence A B A D
t_y = T.ivector('y')  # eg A B A D /s  for sequence A B A D

# build specific rnn type
rnn = None
if opts.type == "simple":
    rnn = SimpleRnn(n_in, n_hidden)
elif opts.type == "bidirectional":
    rnn = BidirectionalRnn(n_in, n_hidden)
elif opts.type == "gru":
    rnn = GruRnn(n_in, n_hidden)
elif opts.type == "attention":
    rnn = AttentionRnn(n_in, n_hidden)
else:
    raise "unknown rnn type? [%s]" % opts.type

# calculate y based on x and initial hidden state of 0
# note for rnns that don't support glimpses the value returned for glimpses will be h0
# TODO this returning h0 is super clusmy, need to fix the API
t_h0 = theano.shared(np.zeros(n_hidden, dtype='float32'), name='h0', borrow=True)
t_y_softmax, glimpses = rnn.t_y_softmax(t_x, t_h0)

# loss is just cross entropy of the softmax output compared to the target
cross_entropy = T.mean(T.nnet.categorical_crossentropy(t_y_softmax, t_y))

# gradient update; either vanilla or rmsprop
learning_rate = 0.05

def vanilla(params, gradients):
    return [(param, param - learning_rate * gradient) for param, gradient in zip(params, gradients)]

def rmsprop(params, gradients):
    updates = []
    for param_t0, gradient in zip(params, gradients):
        # rmsprop see slide 29 of http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        # first the mean_sqr exponential moving average
        mean_sqr_t0 = theano.shared(np.zeros(param_t0.get_value().shape, dtype=param_t0.get_value().dtype))  # zeros in same shape are param
        mean_sqr_t1 = 0.9 * mean_sqr_t0 + 0.1 * (gradient**2)
        updates.append((mean_sqr_t0, mean_sqr_t1))
        # update param surpressing gradient by this average
        param_t1 = param_t0 - learning_rate * (gradient / T.sqrt(mean_sqr_t1 + 1e-10))
        updates.append((param_t0, param_t1))
    return updates

# lookup update fn from opts
update_fn = globals().get(opts.adaptive_learning_rate_fn)
if update_fn == None:
    raise Exception("no update_fn " + opts.adaptive_learning_rate_fn)

gradients = T.grad(cost=cross_entropy, wrt=rnn.params())
updates = update_fn(rnn.params(), gradients)


compile_start_time = time.time()

# compile function for training; ie with backprop updates
train_fn = theano.function(inputs=[t_x, t_y],
                           outputs=[cross_entropy],
                           updates=updates)

# compile function to emit predictions
predict_fn = theano.function(inputs=[t_x],
                             outputs=[t_y_softmax, glimpses])


print "compilation took %0.3f s" % (time.time()-compile_start_time)

for epoch in range(opts.epochs):
    start_time = time.time()

    # train on 1000 examples. no batching yet!! o_O
    for _ in xrange(1000):
        training_eg = rb.ids_for(rb.embedded_reber_sequence())
        x, y = training_eg[:-1], training_eg[1:]
        cost, = train_fn(x, y)

    # test on more, for now just 1 since we're hacking
    # with glimpse vectors
    prob_seqs = []
    for test_idx in xrange(1):
        probabilities = []
        test_eg = rb.ids_for(rb.embedded_reber_sequence())
        x, y = test_eg[:-1], test_eg[1:]
        print "test_idx", test_idx
        print "x", rb.tokens_for(x)
        print "y", rb.tokens_for(y)

        y_softmaxs, glimpses = predict_fn(x)

        # need to have glimpses len match y for the zip so swap
        # h0 "marker" for [None]
        # OMG such clumsy, much hack, terrible API.
        if glimpses[0].__class__ == np.float32:
            glimpses = [None] * len(y)

        for n, (y_true_i, y_softmax, glimpse) in enumerate(zip(y, y_softmaxs, glimpses)):
            print "(%s) -> (%s)" % (rb.LABELS[x[n]], rb.LABELS[y_true_i])
            print "  y_softmax", zip(rb.LABELS, util.float_array_to_str(y_softmax))
            if glimpse is not None:
                print "  glimpse", zip(rb.tokens_for(x), util.float_array_to_str(glimpse))
            y_true_confidence = y_softmax[y_true_i]
            probabilities.append(y_true_confidence)
        prob_seqs.append(probabilities)

    print "epoch", epoch,
    print util.perplexity_stats(prob_seqs),
    print "took %.3f sec" % (time.time()-start_time)
