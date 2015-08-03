#!/usr/bin/env python

# trivial (testing) perfect model; predicts sequence perfectly
# expect perplexity == 1.0

import sys
import numpy as np
import util
import reber_grammar as rb

# training; just record set of observed symbols
# ignore data

# test; assume perfect prediction
prob_seqs = []
for _ in xrange(100):
    seq = rb.embedded_reber_sequence(include_start_end=False)
    prob_seqs.append([1.0] * len(seq))
print util.perplexity_stats(prob_seqs)
