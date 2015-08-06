#!/usr/bin/env python
import numpy as np
import util
import theano
import theano.tensor as T

class SimpleRnn(object):
    def __init__(self, n_in, n_embedding, n_hidden):
        self.Wx = util.sharedMatrix(n_in, n_embedding, 'Wx')
        self.Wrec = util.sharedMatrix(n_hidden, n_embedding, 'Wrec')
        self.Wy = util.sharedMatrix(n_in, n_hidden, 'Wy')

    def params(self):
        return [self.Wx, self.Wrec, self.Wy]

    def recurrent_step(self, x_t, h_t_minus_1):
        # calc new hidden state; elementwise add of embedded input & 
        # recurrent weights dot _last_ hiddenstate
        embedding = self.Wx[x_t]
        h_t = T.tanh(h_t_minus_1 + T.dot(self.Wrec, embedding))

        # calc output; softmax over output weights dot hidden state
        y_t = T.flatten(T.nnet.softmax(T.dot(self.Wy, h_t)), 1)

        # return what we want to have per output step
        return [h_t, y_t]

    def t_y_softmax(self, x, h0):
        [_hs, y_softmax], _ = theano.scan(fn=self.recurrent_step,
                                             sequences=[x],
                                             outputs_info=[h0, None])
        # we return h0 to denote no additional debugging info supplied with softmax
        # TODO fix this; super clumsy api
        return y_softmax, h0
