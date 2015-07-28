#!/usr/bin/env python
import numpy as np
import util
import theano
import theano.tensor as T

class GruRnn(object):
    def __init__(self, n_in, n_hidden):
        self.Wx = util.sharedMatrix(n_hidden, n_in, 'Wx')
        self.Wz = util.sharedMatrix(n_hidden, n_in, 'Wz')
        self.Wr = util.sharedMatrix(n_hidden, n_in, 'Wr')
        self.Ux = util.sharedMatrix(n_hidden, n_hidden, 'Ux')
        self.Uz = util.sharedMatrix(n_hidden, n_hidden, 'Uz')
        self.Ur = util.sharedMatrix(n_hidden, n_hidden, 'Ur')
        self.Wy = util.sharedMatrix(n_in, n_hidden, 'Wy')

    def params(self):
        return [self.Wx, self.Wz, self.Wr, self.Ux, self.Uz, self.Ur, self.Wy]

    def recurrent_step(self, x_t, h_t_minus_1):
        # calc reset gate activation
        r = T.nnet.sigmoid(self.Wr[:, x_t] + T.dot(self.Ur, h_t_minus_1))
        # calc candidate next hidden state (with reset)
        embedding = self.Wx[:, x_t]
        h_t_candidate = T.tanh(embedding + T.dot(self.Ux, r * h_t_minus_1))

        # calc update gate activation 
        z = T.nnet.sigmoid(self.Wz[:, x_t] + T.dot(self.Uz, h_t_minus_1))
        # calc hidden state as affine combo of last state and candidate next state
        h_t = (1 - z) * h_t_minus_1 + z * h_t_candidate

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
