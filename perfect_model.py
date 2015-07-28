#!/usr/bin/env python

# trivial (testing) perfect model; predicts sequence perfectly
# expect perplexity == 1.0

import sys
import numpy as np
from util import perplexities_and_second_last_probs
import reber_grammar as rb

# training; just record set of observed symbols
# ignore data

# test; assume perfect prediction
prob_seqs = []
for _ in xrange(100):
    seq = rb.embedded_reber_sequence(include_start_end=False)
    prob_seqs.append([1.0] * len(seq))
print perplexities_and_second_last_probs(prob_seqs)
