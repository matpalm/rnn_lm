#!/usr/bin/env python

# trivial uniform model
# expect perplexity == |vocab|

import sys
import numpy as np
import util
import reber_grammar as rb

# training; just record set of observed symbols
observed_symbols = set()
for _ in xrange(1000):    
    seq = rb.embedded_reber_sequence()
    observed_symbols.update(seq)

# test; probability of anything in sequence is uniform
prob_seqs = []
uniform_prob = 1.0 / len(observed_symbols)
for _ in xrange(100):
    seq = rb.embedded_reber_sequence()
    prob_seqs.append([uniform_prob] * len(seq))
print util.perplexity_stats(prob_seqs)
