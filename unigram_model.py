#!/usr/bin/env python

# unigram model
import sys
import numpy as np
from collections import defaultdict
import util
import reber_grammar as rb

# training; record frequencies per symbol, later used to determine P(w)
token_freq = defaultdict(int)
total = 0
for _ in xrange(1000):
    seq = rb.embedded_reber_sequence(include_start_end=False)
    for c in seq:
        token_freq[c] += 1
    total += len(seq)

# test; probability of anything is based on unigram frequency
prob_seqs = []
for _ in xrange(100):
    seq = rb.embedded_reber_sequence(include_start_end=False)
    probs = [float(token_freq[c]) / total for c in seq]    
    prob_seqs.append(probs)
print util.perplexity_stats(prob_seqs)
