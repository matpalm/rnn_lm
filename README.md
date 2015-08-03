# RNN language model hacking

## reber grammar

the [reber grammar](http://www.willamette.edu/~gorr/classes/cs449/reber.html) is a old standard 
for RNN testing. in particular we'll use the embedded form of the grammar.

```
$ ./reber_grammar.py | head -n5 
BPBPTVVEPE
BPBTSSSXXVVEPE
BTBTSSXXVPXVPXVPXTTTTVVETE
BPBPVPSEPE
BTBPTVVETE
```

one interesting thing to note in the embedded form of the grammar is that the second token is
always the same as the second last token; either a P or a T. this is one of the long term dependencies the
model needs to learn to handle, though we'll see it's trivial for some models.

lengths of string are potentially unbounded but majority are <20
(histogram.py provided by the awesome [data_hacks](https://github.com/bitly/data_hacks) lib)

```
$ ./reber_grammar.py 100000 | perl -ne'print length($_)."\n";' | histogram.py
# NumSamples = 100000; Min = 10.00; Max = 45.00
# Mean = 13.007970; Variance = 11.508306; SD = 3.392389; Median 12.000000
# each ∎ represents a count of 896
   10.0000 -    13.5000 [ 67254]: ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
   13.5000 -    17.0000 [ 22887]: ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
   17.0000 -    20.5000 [  5923]: ∎∎∎∎∎∎
   20.5000 -    24.0000 [  2766]: ∎∎∎
   24.0000 -    27.5000 [   702]: 
   27.5000 -    31.0000 [   325]: 
   31.0000 -    34.5000 [    93]: 
   34.5000 -    38.0000 [    37]: 
   38.0000 -    41.5000 [     8]: 
   41.5000 -    45.0000 [     5]: 
```

we'll examine a number of differing models for this task and report two stats
1) [perplexity](http://en.wikipedia.org/wiki/Perplexity#Perplexity_per_word)
and 2) the precision of predicting the second last character (a network should
be able to 'remember' this long range dependency) note though that we report
the third last character since all training includes a synthetic start (<s>)
and end (</s>) of sentence token.

## trivial models

### sanity check models

just included as a sanity check stats

```
# just assume P(w) is uniform (grammar has 9 items; 1/9 = 0.111)
$ ./uniform_model.py
min, mean, max  perplexity (7.000 7.000 7.000)  third_last (0.111 0.111 0.111)

# perfect model predicts every transistion perfectly
$ ./perfect_model.py
min, mean, max  perplexity (1.000 1.000 1.000)  third_last (1.000 1.000 1.000)
```

### unigram model

P(w_{n} | w_{n-1})

not much better than just a uniform model. 
terrible at the second last prediction since it's just the frequency of the observed tokens.

```
$ ./unigram_model.py
min, mean, max  perplexity (5.844 6.726 8.072)  third_last (0.164 0.181 0.209)
```

### bigram model

P(w_{n} | w_{n-1}, W_{n-2})

```
$ ./bigram_model.py
min, mean, max  perplexity (2.742 3.128 3.933)  third_last (0.495 0.499 0.505)
```

### some rnns

![cost](cost.png?raw=true "cost")

#### v1. simple as you can get

* single layer RNN
* no gating within unit at all
* no adaptive learning rates / schedules, just fixed rate
* no batching, train one example at a time.
* trivial randn weight init
* no bias with dot products

can see a much lower perplexity compares to ngram model but worse performance at second_last
performance

```
$ ./rnn.py --type=simple --adaptive-learning-rate=vanilla
compilation took 6.698 s
epoch 0 min, mean, max  perplexity (3.354 4.734 7.106)  third_last (0.091 0.234 0.354) took 0.986 sec
epoch 1 min, mean, max  perplexity (2.360 3.137 5.002)  third_last (0.175 0.343 0.466) took 0.950 sec
epoch 2 min, mean, max  perplexity (1.920 2.547 3.892)  third_last (0.249 0.421 0.557) took 0.966 sec
epoch 3 min, mean, max  perplexity (1.803 2.152 3.051)  third_last (0.355 0.440 0.525) took 0.954 sec
epoch 4 min, mean, max  perplexity (1.690 2.018 2.693)  third_last (0.421 0.466 0.506) took 0.957 sec
```

#### v2. using rmsprop adaptive learning rate

* same as simple but using rmsprop (the default for --adaptive-learning-rate)
* uses twice the parameters as the non rmsprop version (each param has a stored gradient moving average)

main difference to previous model is convergence much faster

```
$ ./rnn.py --type=simple --adaptive-learning-rate=rmsprop
compilation took 6.381 s
epoch 0 min, mean, max  perplexity (1.380 1.571 1.897)  third_last (0.401 0.505 0.600) took 1.001 sec
epoch 1 min, mean, max  perplexity (1.416 1.787 3.383)  third_last (0.322 0.507 0.782) took 0.978 sec
epoch 2 min, mean, max  perplexity (1.299 1.574 1.929)  third_last (0.191 0.601 0.939) took 0.999 sec
epoch 3 min, mean, max  perplexity (1.259 1.559 2.212)  third_last (0.052 0.653 0.976) took 1.001 sec
epoch 4 min, mean, max  perplexity (1.281 1.576 2.103)  third_last (0.115 0.696 0.999) took 0.974 sec
```

#### v3. bidirectional rnn

* same as simple rnn but with bidirectional layer
* uses twice the parameters again; needs Wx, Wrec & Wy for _both_ directions

occasionally gets the sequence perfect (but this is luck since the generator is stochastic) but is
immediately perfect now at third_last (makes sense given the shared forward/backwards features)

```
$ ./rnn.py --type=bidirectional
compilation took 18.921 s
epoch 0 min, mean, max  perplexity (1.003 1.190 1.798)  third_last (0.980 0.998 1.000) took 1.892 sec
epoch 1 min, mean, max  perplexity (1.001 1.217 2.298)  third_last (0.999 1.000 1.000) took 1.945 sec
epoch 2 min, mean, max  perplexity (1.000 1.161 1.863)  third_last (1.000 1.000 1.000) took 1.960 sec
epoch 3 min, mean, max  perplexity (1.000 1.194 2.955)  third_last (1.000 1.000 1.000) took 1.910 sec
epoch 4 min, mean, max  perplexity (1.000 1.134 1.679)  third_last (1.000 1.000 1.000) took 1.898 sec
```

#### v4. gru

* same as simple (unidirectional) but this time with [GRU](http://arxiv.org/abs/1502.02367)

seems to learn the long term dependency, eventually... (though sometimes it's immediately)

```
$ ./rnn.py --type=gru
compilation took 11.123 s
epoch 0 min, mean, max  perplexity (1.256 1.615 2.045)  third_last (0.264 0.498 0.733) took 1.699 sec
epoch 1 min, mean, max  perplexity (1.386 1.595 2.060)  third_last (0.435 0.458 0.561) took 1.688 sec
epoch 2 min, mean, max  perplexity (1.305 1.589 2.282)  third_last (0.323 0.412 0.676) took 1.741 sec
epoch 3 min, mean, max  perplexity (1.329 1.576 1.920)  third_last (0.385 0.400 0.605) took 1.722 sec
epoch 4 min, mean, max  perplexity (1.473 1.540 1.838)  third_last (0.434 0.561 0.564) took 1.707 sec
epoch 5 min, mean, max  perplexity (1.390 1.558 1.972)  third_last (0.407 0.413 0.592) took 1.722 sec
epoch 6 min, mean, max  perplexity (1.379 1.546 2.067)  third_last (0.485 0.489 0.513) took 1.711 sec
epoch 7 min, mean, max  perplexity (1.270 1.662 2.092)  third_last (0.257 0.272 0.743) took 1.723 sec
epoch 8 min, mean, max  perplexity (1.369 1.594 1.940)  third_last (0.470 0.482 0.528) took 1.709 sec
epoch 9 min, mean, max  perplexity (1.382 1.565 2.137)  third_last (0.428 0.569 0.706) took 1.726 sec
epoch 10 min, mean, max  perplexity (1.300 1.505 1.934)  third_last (0.686 0.992 1.000) took 1.738 sec
epoch 11 min, mean, max  perplexity (1.220 1.527 2.306)  third_last (0.987 0.998 1.000) took 1.734 sec
epoch 12 min, mean, max  perplexity (1.315 1.474 2.001)  third_last (0.746 1.000 1.000) took 1.730 sec
epoch 13 min, mean, max  perplexity (1.252 1.508 2.291)  third_last (0.997 0.999 1.000) took 1.747 sec
epoch 14 min, mean, max  perplexity (1.265 1.462 2.115)  third_last (0.996 1.000 1.000) took 1.734 sec
epoch 15 min, mean, max  perplexity (1.303 1.476 1.698)  third_last (0.996 1.000 1.000) took 1.729 sec
epoch 16 min, mean, max  perplexity (1.296 1.466 1.990)  third_last (1.000 1.000 1.000) took 1.685 sec
epoch 17 min, mean, max  perplexity (1.277 1.478 2.134)  third_last (0.996 1.000 1.000) took 1.717 sec
epoch 18 min, mean, max  perplexity (1.207 1.429 2.581)  third_last (0.999 1.000 1.000) took 1.694 sec
epoch 19 min, mean, max  perplexity (1.253 1.512 2.069)  third_last (0.999 1.000 1.000) took 1.720 sec
```

#### v5. attention

need to write this one up; some interesting results

```
$ ./rnn.py --type=attention
```

## conclusions

all works, but clearly need a harder problem :/
