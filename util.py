import sys
import numpy as np
import theano

def perplexity_of_sequence(probabilities):
    perplexity_sum = sum([np.log2(max(1e-10, p)) for p in probabilities])
    return 2**((-1./len(probabilities)) * perplexity_sum)

def stats(values):
    return map(float, ["%0.3f" % v for v in [np.min(values), np.median(values), np.max(values)]])

def perplexity_stats(prob_seqs):
    if len(prob_seqs) == 0:
        return []
    return stats([perplexity_of_sequence(prob_seq) for prob_seq in prob_seqs])

def third_last_stats(prob_seqs):
    if len(prob_seqs) == 0:
        return []
    return stats([prob_seq[-3] for prob_seq in prob_seqs])

def prob_stats(x, y, probs):
    probs_str = ["%.2f" % p for p in probs]
    return "xyp " + " ".join(map(str, zip(x, y, probs_str)))

def sharedMatrix(n_rows, n_cols, name, orthogonal_init=True):
    if orthogonal_init and n_rows < n_cols:
        print >>sys.stderr, "warning: can't do orthogonal init of %s, since n_rows (%s) < n_cols (%s)" % (name, n_rows, n_cols)
        orthogonal_init = False
    w = np.random.randn(n_rows, n_cols)
    if orthogonal_init:
        w, _s, _v = np.linalg.svd(w, full_matrices=False)
    return theano.shared(np.asarray(w, dtype='float32'), name=name, borrow=True)

def sharedVector(n_elems, name):
    return theano.shared(np.asarray(np.random.randn(n_elems), dtype='float32'), name=name, borrow=True)

def float_array_to_str(ns, sd=2):
    format_string = "%%.0%df" % sd
    return [format_string % n for n in ns]

