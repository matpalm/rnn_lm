#!/usr/bin/env python

# bigram model

import sys
import numpy as np
from collections import defaultdict
import util
import reber_grammar as rb

# training; calculate bigram probabilities P(w2|w1)
bigram_freq = defaultdict(lambda: defaultdict(int))
for _ in xrange(1000):
    seq = rb.embedded_reber_sequence(include_start_end=True)  # to model start/end
    for i in xrange(1, len(seq)):
        bigram_freq[seq[i-1]][seq[i]] += 1
bigram_prob = {}
for w1 in bigram_freq.keys():
    bigram_prob[w1] = {}
    w2_freqs = bigram_freq[w1]
    w2_total = float(sum(w2_freqs.values()))
    for w2, freq in w2_freqs.iteritems():
        bigram_prob[w1][w2] = float(bigram_freq[w1][w2]) / w2_total

# test
prob_seqs = []
for _ in xrange(100):
    seq = rb.embedded_reber_sequence(include_start_end=True)
    # small vocab so assume all bigrams have data from training (ie no need to backoff to unigram model)
    bigram_probabilities = [bigram_prob[seq[i-1]][seq[i]] for i in xrange(1, len(seq))]
    prob_seqs.append(bigram_probabilities)
print util.perplexity_stats(prob_seqs)
