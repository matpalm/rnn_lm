#!/usr/bin/env python

# generate a sequence of embedded reber string
# see original LSTM paper, s5.1 exp1

import random

def coin_flip():
    return 0 if random.random() < 0.5 else 1

def reber_sequence():
    transistions = { 0: [(1, "T"), (3, "P")],
                     1: [(1, "S"), (2, "X")],
                     2: [(3, "X"), (5, "S")],
                     3: [(3, "T"), (4, "V")],
                     4: [(2, "P"), (5, "V")] }
    state, s = 0, ["B"]
    while state != 5:
        next_state, emitted_symbol = transistions[state][coin_flip()]
        state = next_state
        s.append(emitted_symbol)
    s.append("E")
    return s

def embedded_reber_sequence(include_start_end=True):
    path = "T" if coin_flip() else "P"
    embedded_sequence = ["B"] + [path] + reber_sequence() + [path] + ["E"]
    if include_start_end:
        embedded_sequence = ["<s>"] + embedded_sequence + ["</s>"]
    return embedded_sequence

LABELS = ['<s>','B','E','T','P','S','X','V','</s>']
VOCAB_C_N = dict([(c, n) for n, c in enumerate(LABELS)])
VOCAB_N_C = dict([(n, c) for c, n in VOCAB_C_N.iteritems()])

def ids_for(tokens):
    return [VOCAB_C_N[t] for t in tokens]

def tokens_for(ids):
    return [VOCAB_N_C[n] for n in ids]

def vocab_size():
    return len(VOCAB_C_N)

if __name__ == "__main__":
    import sys
    num_to_generate = int(sys.argv[1]) if len(sys.argv) == 2 else 10
    for _ in xrange(0, num_to_generate):
        #print " ".join(map(str, ids_for(embedded_reber_sequence(include_start_end=False))))
        print "".join(embedded_reber_sequence(include_start_end=False))

