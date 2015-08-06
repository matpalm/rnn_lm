#!/usr/bin/env python
import numpy as np
import util
import theano
import theano.tensor as T

class BidirectionalRnn(object):
    def __init__(self, n_in, n_embedding, n_hidden):
        # forward pass
        self.Wx_f = util.sharedMatrix(n_in, n_embedding, 'Wx_f')
        self.Wrec_f = util.sharedMatrix(n_hidden, n_embedding, 'Wrec_f')
        self.Wy_f = util.sharedMatrix(n_in, n_hidden, 'Wy_f')
        # backwards pass
        self.Wx_b = util.sharedMatrix(n_in, n_embedding, 'Wx_b')
        self.Wrec_b = util.sharedMatrix(n_hidden, n_embedding, 'Wrec_b')
        self.Wy_b = util.sharedMatrix(n_in, n_hidden, 'Wy_b')

    def params(self):
        return [self.Wx_f, self.Wrec_f, self.Wy_f, self.Wx_b, self.Wrec_b, self.Wy_b]

    def scan_through_x(self, 
                       x_t,            # sequence to scan
                       h_t_minus_1,    # recurrent state
                       Wx, Wrec, Wy):  # non_sequences
        # calc new hidden state; elementwise add of embedded input &
        # recurrent weights dot _last_ hiddenstate
        embedding = Wx[x_t]
        h_t = T.tanh(h_t_minus_1 + T.dot(Wrec, embedding))

        # calc contribution to y
        y_t = T.dot(Wy, h_t)

        # return next hidden state and y contribution
        return [h_t, y_t]

    def t_y_softmax(self, t_x, h0):
        # forward scan through x collecting contributions to y
        [_h_ts, y_ts_f], _ = theano.scan(fn = self.scan_through_x,
                                         go_backwards = False,
                                         sequences = [t_x],
                                         non_sequences = [self.Wx_f, self.Wrec_f, self.Wy_f],
                                         outputs_info = [h0, None])
        # backwards scan through x collecting contributions to y
        [_h_ts, y_ts_b], _ = theano.scan(fn = self.scan_through_x,
                                         go_backwards = True,
                                         sequences = [t_x],
                                         non_sequences = [self.Wx_b, self.Wrec_b, self.Wy_b],
                                         outputs_info = [h0, None])
        # elementwise combine y contributions and apply softmax
        y_softmax, _ = theano.scan(fn = lambda y_f, y_b: T.flatten(T.nnet.softmax(y_f + y_b), 1),
                                   sequences = [y_ts_f, y_ts_b],
                                   outputs_info = [None])
        # we return h0 to denote no additional debugging info supplied with softmax
        # TODO fix this; super clumsy api
        return y_softmax, h0
