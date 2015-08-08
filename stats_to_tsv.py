#!/usr/bin/env python
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    e = str(d['epoch'])
    # dump each_explicit norm
    grad_norms = d['sample_gradient_l2_norms']
    for k, v in grad_norms.iteritems():
        print "\t".join([e, "norm", k, str(v)])
    # medians of others
    for p in ['costs', 'perplexity', '3rd_last']:
        print "\t".join([e, "stat", p, str(d[p][1])])
