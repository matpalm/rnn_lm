#!/usr/bin/env python
import numpy as np
import util
import theano
import theano.tensor as T

class AttentionRnn(object):
    def __init__(self, n_in, n_embedding, n_hidden, orthogonal_init):
        # for trivial annotation network; both _f (forward) and _b (backwards)
        self.Wx_a_f = util.sharedMatrix(n_in, n_embedding, 'Wx_a_f', orthogonal_init)  # embeddings for annotations
        self.Whx_f = util.sharedMatrix(n_hidden, n_embedding, 'Whx_f', orthogonal_init)
        self.Wx_a_b = util.sharedMatrix(n_in, n_embedding, 'Wx_a_b', orthogonal_init)  # embeddings for annotations
        self.Whx_b = util.sharedMatrix(n_hidden, n_embedding, 'Whx_b', orthogonal_init)
        # for attention network
        self.Wx_g = util.sharedMatrix(n_in, n_embedding, 'Wx_g', orthogonal_init)  # embeddings for glimpses
        self.Wug = util.sharedMatrix(n_hidden, n_embedding, 'Wug', orthogonal_init)
        self.Wag = util.sharedMatrix(n_hidden, n_hidden, 'Wag', orthogonal_init)
        self.wgs = util.sharedVector(n_hidden, 'Wgs')
        # final mapping to y
        self.Wy = util.sharedMatrix(n_in, n_hidden, 'Wy', orthogonal_init)

    def params(self):
        return [self.Wx_a_f, self.Whx_f,
                self.Wx_a_b, self.Whx_b,
                self.Wx_g, self.Wag, self.Wug, self.wgs,
                self.Wy]

    def _annotation_step(self,
                         x_t,          # sequence to scan
                         h_t_minus_1,  # recurrent state
                         Wx_a, Whx):   # non sequences
        # calc new hidden state; elementwise add of embedded input &
        # recurrent weights dot last hiddenstate
        embedding = Wx_a[x_t]
        h_t = T.tanh(h_t_minus_1 + T.dot(Whx, embedding))
        # TODO annotation_t = some_f(h_t) ?
        return [h_t, h_t]

    def _attended_annotation(self, u, annotations):
        # first we need to mix the annotations using 'u' as a the context of
        # attention. we'll be doing _all_ annotations wrt u in one hit, so we
        # need a column broadcastable version of u
        embedding = self.Wx_g[u]
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
        # simple example it's just a forward/backwards pass of a simple RNN concatenated
        [forward_annotations, _hidden], _ = theano.scan(fn=self._annotation_step,
                                                        go_backwards=False,
                                                        sequences=[x],
                                                        non_sequences=[self.Wx_a_f, self.Whx_f],
                                                        outputs_info=[None, h0])
        [backwards_annotations, _hidden], _ = theano.scan(fn=self._annotation_step,
                                                          go_backwards=True,
                                                          sequences=[x],
                                                          non_sequences=[self.Wx_a_b, self.Whx_b],
                                                          outputs_info=[None, h0])
        backwards_annotations = backwards_annotations[::-1]  # to make indexing same as forwards_
        annotations = T.concatenate([forward_annotations, backwards_annotations])

        # second pass; calculate attention over annotations
        # NOTE! there is specifically NO recursion here, just each x wrt the annotations.
        #       this is quite crippling since each token in sequence must be considered
        #       independently of what has happened before.
        [attended_annotations, glimpses], _ = theano.scan(fn=self._attended_annotation,
                                                          sequences=[x],
                                                          non_sequences=[annotations])

        # final pass; apply softmax
        y_softmax, _ = theano.scan(fn=self._softmax,
                                   sequences=[attended_annotations])
        return y_softmax, glimpses
