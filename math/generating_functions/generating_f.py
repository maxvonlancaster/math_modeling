import numpy as np
import matplotlib.pyplot as plt
import math

def _function(t):
    return 1/(1-t-t*t)

def _generating_function(sequence, t):
    res = 0
    for i in range(len(sequence)):
        res += sequence[i] * pow(t, i)
    return res

def _sequence_rule(sequence):
    if len(sequence) == 0:
        return 1
    if len(sequence) == 1:
        return 1
    return sequence[len(sequence) - 1] + sequence[len(sequence) - 2]

def _get_sequence(n):
    sequence = []
    for i in range(n):
        sequence.append(_sequence_rule(sequence))
    return sequence

seq = _get_sequence(20)
print(seq)

# plot a generating function:
a1 = np.arange(0.0, 0.6, 0.001)
a2 = np.arange(0.0, 0.6, 0.001)

plt.figure()
plt.subplot(211)
plt.plot(a1, _generating_function(seq, a1), 'bo')

plt.subplot(212)
plt.plot(a2, _function(a2), '--r')

plt.show()

