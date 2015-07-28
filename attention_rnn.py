#!/usr/bin/env python
import numpy as np
import util
import theano
import theano.tensor as T

class AttentionRnn(object):
    def __init__(self, n_in, n_hidden):
        # for trivial annotation network
        self.Wx = util.sharedMatrix(n_hidden, n_in, 'Wx')  # embeddings
        self.Whx = util.sharedMatrix(n_hidden, n_hidden, 'Whx')
        # for attention network
        self.Wag = util.sharedMatrix(n_hidden, n_hidden, 'Wag')
        self.Wug = util.sharedMatrix(n_hidden, n_hidden, 'Wug')
        self.wgs = util.sharedVector(n_hidden, 'Wgs')
        # final mapping to y
        self.Wy = util.sharedMatrix(n_in, n_hidden, 'Wy')

    def params(self):
        return [self.Wx, self.Whx, self.Wag, self.Wug, self.wgs, self.Wy]

    def _annotation_step(self, x_t, h_t_minus_1):
        # calc new hidden state; elementwise add of embedded input &
        # recurrent weights dot last hiddenstate
        embedding = self.Wx[:, x_t]
        h_t = T.tanh(embedding + T.dot(self.Whx, h_t_minus_1))
        # TODO annotation_t = some_f(h_t) ?
        return [h_t, h_t]

    def _attended_annotation(self, u, annotations):
        # first we need to mix the annotations using 'u' as a the context of
        # attention. we'll be doing _all_ annotations wrt u in one hit, so we
        # need a column broadcastable version of u
        embedding = self.Wx[:, u]
        u_col = embedding.dimshuffle(0, 'x')
        glimpse_vectors = T.tanh(T.dot(self.Wag, annotations.T) + T.dot(self.Wug, u_col))

        # now collapse the glimpse vectors (there's one per token) to scalars
        unnormalised_glimpse_scalars = T.dot(self.wgs, glimpse_vectors)

        # normalise glimpses with a softmax
        exp_glimpses = T.exp(unnormalised_glimpse_scalars)
        glimpses = exp_glimpses / T.sum(exp_glimpses)

        # attended version of the annotations is the the affine combo of the
        # annotations using the normalised glimpses as the combo weights
        attended_annotations = T.dot(annotations.T, glimpses)
        return [attended_annotations, glimpses]

    def _softmax(self, annotation):
        # calc output; softmax over output weights dot hidden state
        return T.flatten(T.nnet.softmax(T.dot(self.Wy, annotation)), 1)

    def t_y_softmax(self, x, h0):
        # first pass is building base annotation vectors. for this
        # simple example it's just a forward pass of a simple RNN
        # but, since annotations define context, it makes more sense
        # to use a bidirectional net here.
        [annotations, _hidden], _ = theano.scan(fn=self._annotation_step,
                                                sequences=[x],
                                                outputs_info=[None, h0])

        # second pass; calculate annotations
        [attended_annotations, glimpses], _ = theano.scan(fn=self._attended_annotation,
                                                          sequences=[x],
                                                          non_sequences=[annotations])

        # final pass; apply softmax
        y_softmax, _ = theano.scan(fn=self._softmax,
                                   sequences=[attended_annotations])
        return y_softmax, glimpses
